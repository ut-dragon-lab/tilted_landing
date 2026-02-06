#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry # Added missing import
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub
from aerial_robot_planning.trajs import BaseTraj

# CONFIGURATION
DESCENT_SPEED = 0.15
# Beetle CoG is ~11.5cm. 0.13m triggers pressing just before physical contact.
PRESS_THRESHOLD = 0.13 
PRESS_TARGET_DIST = 0.10
PRESS_DURATION = 2.0 # Seconds to hold pressure before cutting motors

class TagObserver:
    """
    Handles visual perception by tracking AprilTag frames and applying a 
    Low Pass Filter (LPF) to smooth the data.
    """
    def __init__(self, bundle_frame="land_mark_bundle", small_tag_frame="small_tag"):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.bundle_frame = bundle_frame
        self.small_tag_frame = small_tag_frame
        
        self.lpf_alpha = 0.6 
        self.smooth_pos = np.zeros(3)
        self.last_valid_quat = [0, 0, 0, 1]
        self.initialized = False

    def get_smoothed_transform(self):
        # Priority 1: Small tag (for high precision close to surface)
        target = self._check_frame(self.small_tag_frame)
        
        # Priority 2: Bundle (for stability at higher altitudes)
        if target is None:
            target = self._check_frame(self.bundle_frame)
            
        if target is None:
            return None # Visual contact lost

        raw_pos, raw_quat = target
        
        if not self.initialized:
            self.smooth_pos = raw_pos
            self.last_valid_quat = raw_quat
            self.initialized = True
        else:
            # Apply LPF to position to prevent controller jitter
            self.smooth_pos = (1.0 - self.lpf_alpha) * self.smooth_pos + self.lpf_alpha * raw_pos
            self.last_valid_quat = raw_quat

        return self.smooth_pos, self.last_valid_quat

    def _check_frame(self, frame_name):
        try:
            trans = self.tf_buffer.lookup_transform("world", frame_name, rospy.Time(0), rospy.Duration(0.01))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            return pos, quat
        except:
            return None

class LandingTraj(BaseTraj):
    """
    Analytical trajectory for NMPC that descends along the tag's normal vector 
    and initiates a pressing maneuver upon contact.
    """
    def __init__(self, tag_observer, robot_name):
        super().__init__()
        self.tag_obs = tag_observer
        self.robot_name = robot_name
        self.is_pressing = False
        self.press_start_time = None
        self.is_touchdown = False

        self.curr_odom = None
        rospy.Subscriber(f"/{robot_name}/uav/cog/odom", Odometry, self._odom_cb)

    def _odom_cb(self, msg):
        self.curr_odom = msg

    def get_3d_pt(self, t_cal):
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None or self.curr_odom is None:
            return np.zeros(9)

        t_pos, t_quat = tag_data
        r_mat = R.from_quat(t_quat)
        normal_vec = r_mat.apply([0, 0, 1])

        # [cite_start]Calculate perpendicular Z-distance to the ramp surface [cite: 5]
        curr_p = np.array([self.curr_odom.pose.pose.position.x,
                           self.curr_odom.pose.pose.position.y,
                           self.curr_odom.pose.pose.position.z])
        z_dist = np.dot(curr_p - t_pos, normal_vec)

        # Transition logic: Activate Press Maneuver
        if z_dist < PRESS_THRESHOLD and not self.is_pressing:
            rospy.loginfo(">>> CONTACT DETECTED: STARTING PRESS MANEUVER <<<")
            self.is_pressing = True
            self.press_start_time = rospy.Time.now()

        if self.is_pressing:
            # Maintain a target slightly below surface to generate friction
            target_dist = PRESS_TARGET_DIST
            # Cut motors after the drone has settled for 2 seconds
            if (rospy.Time.now() - self.press_start_time).to_sec() > PRESS_DURATION:
                self.is_touchdown = True
        else:
            # Smooth descent along the tag's normal vector
            target_dist = max(PRESS_TARGET_DIST, 1.0 - (DESCENT_SPEED * t_cal))

        pos = t_pos + (normal_vec * target_dist)
        vel = -normal_vec * DESCENT_SPEED
        return np.concatenate([pos, vel, np.zeros(3)])

    def get_3d_orientation(self, t_cal):
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None: return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        _, t_quat = tag_data
        # Return [qw, qx, qy, qz, rates..., accs...] for NMPC
        return [t_quat[3], t_quat[0], t_quat[1], t_quat[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def check_finished(self, t_elapsed):
        return self.is_touchdown

    def get_frame_id(self): return "world"
    def get_child_frame_id(self): return "cog"

if __name__ == "__main__":
    rospy.init_node('beetle_omni_visual_landing')
    robot_name = "beetle1"
    
    # Initialize components
    observer = TagObserver()
    landing_traj = LandingTraj(observer, robot_name)
    
    # MPCTrajPtPub will use LandingTraj to fill the NMPC horizon
    mpc_node = MPCTrajPtPub(robot_name, landing_traj)
    
    halt_pub = rospy.Publisher(f"/{robot_name}/teleop_command/halt", Empty, queue_size=1)
    
    rospy.loginfo("Visual Landing Mission Started")
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        # Watch for the touchdown flag set by the trajectory logic
        if landing_traj.is_touchdown:
            rospy.logwarn("Touchdown stable! Cutting motors.")
            for _ in range(10): halt_pub.publish(Empty())
            break
        rate.sleep()
