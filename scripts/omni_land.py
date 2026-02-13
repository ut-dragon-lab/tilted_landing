#!/usr/bin/env python3
"""
Omni Beetle Visual Landing Mission
Architectural Decision: Model-View-Controller (MVC) Pattern.
- Model: Trajectory classes provide pure mathematical references.
- View: MPCTrajPtPub handles ROS communication and MPC horizon filling.
- Controller: Main loop and TagObserver coordinate sensor fusion, data publishing, and state transitions.
"""

import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion

# Package imports
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub
from aerial_robot_planning.trajs import BaseTraj

# ==========================================================
# CONFIGURATION PARAMETERS (Global Constants)
# ==========================================================
ROBOT_NAME = "beetle1"
APPROACH_SPEED = 0.5        # m/s during Phase 1
DESCENT_SPEED = 0.3         # m/s during Phase 2 (Perpendicular)
LANDING_OFFSET = 0.8        # Distance (meters) to hover above tag before descent
TRANSITION_THRESHOLD = 0.10  # Error tolerance (meters) to switch phases
PRESS_DURATION = 2.5        # Extra time (seconds) to press against the ramp
TOTAL_DESCENT_DIST = 0.8    # Distance to cover during descent
TAG_BUNDLE_FRAME = "land_mark_bundle"
SMALL_TAG_FRAME = "small_tag"

# ==========================================================
# "MODEL": Trajectory Classes
# ==========================================================
class StartPointApproachTraj(BaseTraj):
    """
    Responsibility: Calculate a dynamic omni-navigation path to the offset point.
    """
    def __init__(self, start_xyz, speed=APPROACH_SPEED, d_offset=LANDING_OFFSET):
        super().__init__()
        self.start_xyz = np.array(start_xyz)
        self.speed = speed
        self.d_offset = d_offset
        self.p_tag = np.zeros(3)
        self.q_tag = np.array([0, 0, 0, 1])
        self.initialized = False
        self.normal = np.array([0, 0, 1])

    def update_tag(self, p_tag, q_tag):
        self.p_tag = p_tag
        self.q_tag = q_tag
        self.initialized = True

    def get_3d_pt(self, t):
        if not self.initialized: 
            return np.concatenate([self.start_xyz, np.zeros(6)])
        r_mat = R.from_quat(self.q_tag)
        self.normal = r_mat.apply(np.array([0, 0, 1]))
        target_xyz = self.p_tag + (self.normal * self.d_offset)
        diff = target_xyz - self.start_xyz
        dist_total = np.linalg.norm(diff)
        if dist_total < 0.01: 
            return np.concatenate([target_xyz, np.zeros(6)])
        duration = dist_total / self.speed
        if t < duration:
            direction = diff / dist_total
            pos = self.start_xyz + (direction * self.speed * t)
            vel = direction * self.speed
        else:
            pos = target_xyz
            vel = np.zeros(3)
        return np.concatenate([pos, vel, np.zeros(3)])

    def get_3d_orientation(self, t):
        return [self.q_tag[3], self.q_tag[0], self.q_tag[1], self.q_tag[2], 0, 0, 0, 0, 0, 0]

class PerpendicularLandingTraj(BaseTraj):
    """
    Responsibility: Execute a pure perpendicular descent with frozen reference.
    """
    def __init__(self, p_start, q_start, speed=DESCENT_SPEED):
        super().__init__()
        self.p_start = np.array(p_start)
        self.q_start = np.array(q_start)
        self.speed = speed
        r_mat = R.from_quat(self.q_start)
        self.normal = r_mat.apply(np.array([0, 0, 1]))

    def get_3d_pt(self, t):
        descent_dist = self.speed * t
        pos = self.p_start - (self.normal * descent_dist)
        vel = -self.normal * self.speed
        return np.concatenate([pos, vel, np.zeros(3)])

    def get_3d_orientation(self, t):
        return [self.q_start[3], self.q_start[0], self.q_start[1], self.q_start[2], 0, 0, 0, 0, 0, 0]

# ==========================================================
# "CONTROLLER": TagObserver (Kalman Filter + Telemetry)
# ==========================================================
class TagObserver:
    """
    Responsibility: Filter tag data and publish raw/filtered telemetry.
    """
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Kalman Matrices
        self.state = np.zeros(3)
        self.P = np.eye(3) * 1.0  
        self.Q = np.eye(3) * 1e-5 
        self.R = np.eye(3) * 0.1  
        self.last_valid_quat = [0, 0, 0, 1]
        self.initialized = False

        # Telemetry Publishers
        self.raw_pub = rospy.Publisher(f"/{ROBOT_NAME}/tilted_landing/land_mark/raw", PoseStamped, queue_size=1)
        self.filtered_pub = rospy.Publisher(f"/{ROBOT_NAME}/tilted_landing/land_mark/filtered", PoseStamped, queue_size=1)

    def _update_kalman(self, measurement):
        S = self.P + self.R
        K = np.dot(self.P, np.linalg.inv(S))
        self.state = self.state + np.dot(K, measurement - self.state)
        self.P = (np.eye(3) - K) @ self.P + self.Q

    def _publish_pose(self, publisher, pos, quat):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position = Point(*pos)
        msg.pose.orientation = Quaternion(*quat)
        publisher.publish(msg)

    def get_smoothed_transform(self):
        target = self._lookup(TAG_BUNDLE_FRAME) or self._lookup(SMALL_TAG_FRAME)
        if not target: return None
        
        raw_pos, raw_quat = target
        self._publish_pose(self.raw_pub, raw_pos, raw_quat)

        if not self.initialized:
            self.state, self.last_valid_quat, self.initialized = raw_pos, raw_quat, True
        else:
            self._update_kalman(raw_pos)
            self.last_valid_quat = raw_quat
        
        self._publish_pose(self.filtered_pub, self.state, self.last_valid_quat)
        return self.state, self.last_valid_quat

    def _lookup(self, frame_name):
        try:
            trans = self.tf_buffer.lookup_transform("world", frame_name, rospy.Time(0), rospy.Duration(0.01))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            return pos, quat
        except: return None

# ==========================================================
# MISSION EXECUTION
# ==========================================================
def run_omni_beetle_mission():
    rospy.init_node('omni_beetle_landing_mission')
    observer = TagObserver()
    
    # Status Publisher
    status_pub = rospy.Publisher(f"/{ROBOT_NAME}/tilted_landing/status", String, queue_size=1, latch=True)
    
    # Command Topics (Reference: keyboard_command.py)
    ns = f"/{ROBOT_NAME}/teleop_command"
    force_land_pub = rospy.Publisher(ns + '/force_landing', Empty, queue_size=1)
    # halt_pub = rospy.Publisher(ns + '/halt', Empty, queue_size=1)

    # --- PHASE 1: DYNAMIC APPROACH ---
    status_pub.publish("PHASE_1")
    rospy.loginfo("PHASE 1: Approaching start point...")
    odom = rospy.wait_for_message(f"/{ROBOT_NAME}/uav/cog/odom", Odometry)
    p_init = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]
    
    approach_traj = StartPointApproachTraj(p_init)
    mpc_interface = MPCTrajPtPub(ROBOT_NAME, approach_traj)

    while not rospy.is_shutdown():
        tag = observer.get_smoothed_transform()
        if tag:
            approach_traj.update_tag(tag[0], tag[1])
            
            curr_odom = rospy.wait_for_message(f"/{ROBOT_NAME}/uav/cog/odom", Odometry)
            curr_p = np.array([curr_odom.pose.pose.position.x, curr_odom.pose.pose.position.y, curr_odom.pose.pose.position.z])
            
            target_pt = approach_traj.p_tag + approach_traj.normal * LANDING_OFFSET
            if np.linalg.norm(curr_p - target_pt) < TRANSITION_THRESHOLD:
                rospy.loginfo("Aligned. Transitioning to Perpendicular Descent.")
                break
        rospy.sleep(0.05)

    # --- PHASE 2: PERPENDICULAR DESCENT ---
    status_pub.publish("PHASE_2")
    rospy.loginfo("PHASE 2: Descending...")
    tag_final = observer.get_smoothed_transform()
    landing_model = PerpendicularLandingTraj(target_pt, tag_final[1])
    
    mpc_interface.traj = landing_model 
    mpc_interface.start_time = rospy.Time.now().to_sec()

    # Wait for calculated descent duration + pressure phase
    rospy.sleep((TOTAL_DESCENT_DIST / DESCENT_SPEED) + PRESS_DURATION)
    
    # --- CUTOFF: TRIGGERING LANDING COMMAND ---
    status_pub.publish("CUTOFF")
    rospy.loginfo("Touchdown detected. Triggering force landing.")
    for _ in range(10):
        force_land_pub.publish(Empty())
        # halt_pub.publish(Empty()) # Uncomment to swap force_land with halt
        rospy.sleep(0.1)

    # --- SUCCESS ---
    status_pub.publish("SUCCEDED")
    rospy.loginfo("Mission Accomplished.")

if __name__ == "__main__":
    try:
        run_omni_beetle_mission()
    except rospy.ROSInterruptException:
        pass
