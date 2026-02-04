#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import numpy as np
import random
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty
from aerial_robot_msgs.msg import FlightNav
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist, Vector3, Quaternion
from nav_msgs.msg import Odometry

# Importing the base class for MPC Trajectory Generation
from aerial_robot_planning.pub_mpc_joint_traj import MPCPubJointTraj

# ==============================================================================
# CLASS: TagObserver ( The Eye )
# ==============================================================================
class TagObserver:
    """
    TagObserver is responsible for the 'Vision' aspect of the landing.
    
    PROBLEM: 
    Raw AprilTag detection is noisy. The position jitters, and frames can be dropped.
    Feeding raw data to the controller causes the drone to twitch unstable.

    SOLUTION:
    We implement a Low Pass Filter (LPF) on the tag's position.
    Equation: smoothed_pos = (1 - alpha) * old_pos + alpha * new_pos
    
    - Alpha 0.4: We trust 40% of the new reading and keep 60% of inertia.
                 This removes high-frequency noise while remaining reactive.
    - Memory: If the tag is lost (blind spot), we return the last known valid
              position/orientation instead of getting lost.
    """
    def __init__(self, target_frame="land_mark", world_frame="world"):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.target_frame = target_frame
        self.world_frame = world_frame
        
        self.initialized = False
        self.lpf_alpha = 0.4  # Filter gain (0.4 = Reactive but smooth)
        self.smooth_pos = np.zeros(3)
        self.last_valid_quat = [0, 0, 0, 1] 

    def get_smoothed_transform(self):
        try:
            # Try to get the latest transform with a very short timeout
            trans = self.tf_buffer.lookup_transform(self.world_frame, self.target_frame, rospy.Time(0), rospy.Duration(0.01))
            
            raw_pos = np.array([trans.transform.translation.x, 
                                trans.transform.translation.y, 
                                trans.transform.translation.z])
            
            raw_quat = [trans.transform.rotation.x, 
                        trans.transform.rotation.y, 
                        trans.transform.rotation.z, 
                        trans.transform.rotation.w]

            if not self.initialized:
                self.smooth_pos = raw_pos
                self.last_valid_quat = raw_quat
                self.initialized = True
            else:
                # Apply Low Pass Filter (LPF) to position
                self.smooth_pos = (1.0 - self.lpf_alpha) * self.smooth_pos + self.lpf_alpha * raw_pos
                # For orientation, we just update it (complex filtering on quats is heavy)
                self.last_valid_quat = raw_quat 

            return self.smooth_pos, self.last_valid_quat

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # If visual contact is lost, rely on 'Memory' (last valid state)
            if not self.initialized:
                return None
            return self.smooth_pos, self.last_valid_quat

# ==============================================================================
# CLASS: LandingMPC ( The Brain )
# ==============================================================================
class LandingMPC(MPCPubJointTraj):
    """
    LandingMPC handles the precise control logic during the final descent.
    It inherits from the JSK MPC infrastructure but overrides the trajectory generation.
    
    KEY CONCEPTS:
    1. Vector Math: The drone descends along the TAG'S normal vector, not the World Z.
       This ensures perpendicular landing even on inclined slopes.
    
    2. Pressing Maneuver (Friction Logic):
       - Problem: Gazebo physics causes the drone to slide on slopes if motors stop.
       - Solution: Before cutting motors, we command a target slightly *below* the surface.
       - This generates extra Downward Force (Normal Force), increasing friction.
       - F_friction = mu * N. Higher N -> Drone sticks to the ramp.
       
    3. Performance:
       - Runs at 20Hz (manual timer) to save CPU compared to the default 100Hz.
    """
    def __init__(self, robot_name, tag_observer):
        super().__init__(robot_name, "landing_mpc_node", odom_frame_id="cog")
        
        # --- TIMER CONFIGURATION ---
        # We manually set the timer to 20Hz (0.05s) for optimal performance.
        # Python struggles with complex vector math at higher frequencies.
        self.ts_pt_pub = 0.05 
        self.tmr_pt_pub = rospy.Timer(rospy.Duration(self.ts_pt_pub), self._timer_callback)
        
        self.tag_obs = tag_observer
        self.descent_speed = 0.15 
        
        # --- CONTACT PARAMETERS (User Defined) ---
        # Distance to trigger the 'Pressing' state
        self.trigger_press_threshold = 0.12 
        
        # Target distance during pressing (Pushing into the surface)
        self.press_target_dist = 0.10
        
        # State flags
        self.is_pressing = False
        self.press_start_time = None
        self.is_touchdown = False
        
        # Publishers
        self.halt_pub = rospy.Publisher(f"/{robot_name}/teleop_command/halt", Empty, queue_size=1)
        
        # Pre-allocate trajectory message
        self.traj_msg = MultiDOFJointTrajectory()
        self.traj_msg.header.frame_id = "world"
        self.traj_msg.joint_names.append("cog")
        
        rospy.loginfo(f"{self.namespace}/landing_mpc_node: Initialized with User Parameters.")

    def fill_trajectory_points(self, t_elapsed: float):
        """
        Calculates the NMPC Horizon (prediction steps).
        Called every 0.05s by the timer.
        """
        self.traj_msg.points.clear()
        self.traj_msg.header.stamp = rospy.Time.now()

        # 1. Perception: Get Tag State
        tag_data = self.tag_obs.get_smoothed_transform()
        
        # Safety: Hover if tag is completely invalid and no memory exists
        if tag_data is None:
            if self.uav_odom is not None:
                return self._generate_hover_trajectory_at_current_pose(), None
            return self.traj_msg, None
            
        t_pos, t_quat = tag_data 
        
        # 2. Geometry: Calculate Vectors
        # Convert Quaternion to Rotation Matrix to get the Normal Vector (Z-axis of the tag)
        r_mat = R.from_quat(t_quat)
        normal_vec = r_mat.apply([0, 0, 1]) 
        
        current_pos = np.array([self.uav_odom.pose.pose.position.x,
                                self.uav_odom.pose.pose.position.y,
                                self.uav_odom.pose.pose.position.z])
        
        # Calculate perpendicular distance to the plane (Dot Product)
        dist_to_plane = np.dot(current_pos - t_pos, normal_vec)
        
        # 3. Decision Logic: Contact Detection
        if dist_to_plane < self.trigger_press_threshold and not self.is_pressing:
            rospy.loginfo(">>> CONTACT DETECTED! INITIATING PRESS MANEUVER <<<")
            self.is_pressing = True
            self.press_start_time = rospy.Time.now()

        # 4. Strategy: Define Target
        if self.is_pressing:
            # Force the drone to a fixed point to generate pressure/friction
            future_dist = self.press_target_dist
        else:
            # Standard descent: Reduce distance smoothly
            future_dist = max(self.press_target_dist, dist_to_plane - (self.descent_speed * 0.1))

        # 5. Trajectory Generation: Populate the Horizon
        target_quat_msg = Quaternion(*t_quat)
        desired_pos_world = t_pos + (normal_vec * future_dist)
        linear_vel_vec = Vector3(*(-normal_vec * self.descent_speed))
        zero_twist = Twist() 

        for i in range(self.N_nmpc + 1):
            dt_pred = i * self.T_step
            
            if not self.is_pressing:
                # Approach Curve: Exponential decay or Linear descent
                traj_dist = max(self.press_target_dist, dist_to_plane - (self.descent_speed * dt_pred))
                if dist_to_plane > 1.2:
                     # Smoother approach when far away
                     traj_dist = 1.0 + (dist_to_plane - 1.0) * np.exp(-1.0 * dt_pred)
                p_world = t_pos + (normal_vec * traj_dist)
            else:
                # Lock target during pressing
                p_world = desired_pos_world

            traj_pt = MultiDOFJointTrajectoryPoint()
            trans = Transform()
            trans.translation = Vector3(p_world[0], p_world[1], p_world[2])
            trans.rotation = target_quat_msg 
            
            traj_pt.transforms.append(trans)
            traj_pt.velocities.append(Twist(linear=linear_vel_vec))
            traj_pt.accelerations.append(zero_twist)
            traj_pt.time_from_start = rospy.Duration.from_sec(t_elapsed + dt_pred)
            self.traj_msg.points.append(traj_pt)

        # 6. Termination: Check Stability Time
        if self.is_pressing:
            # Hold the press for 2 seconds before cutting motors
            duration = (rospy.Time.now() - self.press_start_time).to_sec()
            if duration > 2.0:
                self.is_touchdown = True

        return self.traj_msg, None

    def _generate_hover_trajectory_at_current_pose(self):
        """Helper to generate a stationary trajectory (Hover)."""
        self.traj_msg.points.clear()
        curr_p = self.uav_odom.pose.pose.position
        curr_q = self.uav_odom.pose.pose.orientation
        zero_twist = Twist()
        for i in range(self.N_nmpc + 1):
            pt = MultiDOFJointTrajectoryPoint()
            pt.transforms.append(Transform(translation=curr_p, rotation=curr_q))
            pt.velocities.append(zero_twist)
            pt.accelerations.append(zero_twist)
            pt.time_from_start = rospy.Duration.from_sec(i * self.T_step)
            self.traj_msg.points.append(pt)
        return self.traj_msg

    def pub_trajectory_points(self, msg):
        if isinstance(msg, tuple): traj_msg = msg[0]
        else: traj_msg = msg
        if traj_msg is not None and len(traj_msg.points) > 0:
            self.pub_ref_traj.publish(traj_msg)

    def check_finished(self, t_elapsed):
        if self.is_touchdown:
            rospy.logwarn(">>> STABLE. CUTTING MOTORS. <<<")
            for _ in range(5):
                self.halt_pub.publish(Empty())
            return True
        return False

# ==============================================================================
# SMACH STATES ( The Director )
# ==============================================================================

class ArmAndTakeoff(smach.State):
    """
    State: ArmAndTakeoff
    Robust takeoff sequence that checks actual Odometry height before proceeding.
    Avoids 'false starts' where the drone thinks it's flying but is still on the ground.
    """
    def __init__(self, robot_name):
        smach.State.__init__(self, outcomes=['succeeded'])
        self.robot_name = robot_name
        self.start_pub = rospy.Publisher(f"/{robot_name}/teleop_command/start", Empty, queue_size=1)
        self.takeoff_pub = rospy.Publisher(f"/{robot_name}/teleop_command/takeoff", Empty, queue_size=1)
        self.current_z = 0.0
        rospy.Subscriber(f"/{robot_name}/uav/cog/odom", Odometry, self.odom_cb)
        
    def odom_cb(self, msg):
        self.current_z = msg.pose.pose.position.z

    def execute(self, userdata):
        rospy.loginfo("STATE: Arming...")
        for _ in range(3):
            self.start_pub.publish(Empty())
            rospy.sleep(0.2)
        rospy.sleep(1.0)
        
        rospy.loginfo("STATE: Taking Off...")
        timeout = rospy.Time.now() + rospy.Duration(15.0)
        
        # Loop until the drone physically rises above 0.3m
        while rospy.Time.now() < timeout:
            if self.current_z < 0.3:
                self.takeoff_pub.publish(Empty())
                rospy.sleep(0.5)
            else:
                rospy.loginfo(f"Takeoff Detected! Current Z: {self.current_z:.2f}m")
                rospy.sleep(2.0)
                return 'succeeded'
        return 'succeeded'

class RandomizedApproach(smach.State):
    """
    State: RandomizedApproach
    Simulates a 'Macro Navigation' phase (like GPS) with some inherent noise.
    Uses 'Force Feed' control mode to ensure the drone accepts the position target.
    """
    def __init__(self, robot_name):
        smach.State.__init__(self, outcomes=['arrived'])
        self.robot_name = robot_name
        self.nav_pub = rospy.Publisher(f"/{robot_name}/uav/nav", FlightNav, queue_size=1)
        rospy.Subscriber(f"/{robot_name}/uav/cog/odom", Odometry, self.odom_cb)
        self.curr_pos = np.array([0.,0.,0.])

    def odom_cb(self, msg):
        self.curr_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

    def execute(self, userdata):
        target_x = 1.0 + random.uniform(-0.1, 0.1)
        target_y = 1.0 + random.uniform(-0.1, 0.1)
        target_z = 1.5 
        target_vec = np.array([target_x, target_y, target_z])
        rospy.loginfo(f"STATE: Going to ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})...")
        
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        
        # Keep sending commands for 10 seconds or until reached
        while (rospy.Time.now() - start_time).to_sec() < 10.0:
            if rospy.is_shutdown(): return 'arrived'
            
            msg = FlightNav()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.control_frame = 0 
            msg.target = 1        
            msg.pos_xy_nav_mode = 2 
            msg.pos_z_nav_mode = 2
            msg.yaw_nav_mode = 2    
            msg.target_pos_x = target_x
            msg.target_pos_y = target_y
            msg.target_pos_z = target_z
            msg.target_yaw = 0.0 
            
            self.nav_pub.publish(msg)
            
            dist = np.linalg.norm(self.curr_pos - target_vec)
            if dist < 0.2:
                rospy.loginfo("Target Reached! Switching to Landing...")
                return 'arrived'
            rate.sleep()
        return 'arrived'

class VisualServoLanding(smach.State):
    """
    State: VisualServoLanding
    Activates the LandingMPC controller. This state yields control to the MPC
    until the touchdown condition is met.
    """
    def __init__(self, robot_name):
        smach.State.__init__(self, outcomes=['landed', 'aborted'])
        self.robot_name = robot_name
        self.tag_obs = TagObserver()
        
    def execute(self, userdata):
        rospy.loginfo("STATE: Visual Servo Landing Activated!")
        rospy.sleep(0.5)
        
        landing_controller = LandingMPC(self.robot_name, self.tag_obs)
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if landing_controller.is_finished:
                return 'landed'
            rate.sleep()
        return 'aborted'

def main():
    rospy.init_node('ramp_landing_mission')
    robot_name = "beetle1"
    
    sm = smach.StateMachine(outcomes=['MISSION_COMPLETE', 'MISSION_FAILED'])
    with sm:
        smach.StateMachine.add('TAKEOFF', ArmAndTakeoff(robot_name), transitions={'succeeded':'APPROACH'})
        smach.StateMachine.add('APPROACH', RandomizedApproach(robot_name), transitions={'arrived':'LANDING'})
        smach.StateMachine.add('LANDING', VisualServoLanding(robot_name), transitions={'landed':'MISSION_COMPLETE', 'aborted':'MISSION_FAILED'})

    sm.execute()

if __name__ == '__main__':
    main()
