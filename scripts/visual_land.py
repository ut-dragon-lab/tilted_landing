#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Empty
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub
from aerial_robot_planning.trajs import BaseTraj

# CONFIGURATION
DESCENT_SPEED = 0.15
PRESS_THRESHOLD = 0.12
PRESS_TARGET_DIST = 0.10

class TagObserver:
    def __init__(self, bundle_frame="land_mark_bundle", small_tag_frame="small_tag"):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.bundle_frame = bundle_frame
        self.small_tag_frame = small_tag_frame
        
        # Set the LPF
        self.lpf_alpha = 0.6 
        self.smooth_pos = np.zeros(3)
        self.last_valid_quat = [0, 0, 0, 1]
        self.initialized = False

    def get_smoothed_transform(self):
        # First Priority: Attempt small tag (visible only more close to the land platform)
        target = self._check_frame(self.small_tag_frame)
        
        # Second Priority: Check the bundle if the small tag is not visible
        if target is None:
            target = self._check_frame(self.bundle_frame)
            
        if target is None:
            return None # Tag lost

        raw_pos, raw_quat = target
        
        # Apply the LPF
        if not self.initialized:
            self.smooth_pos = raw_pos
            self.last_valid_quat = raw_quat
            self.initialized = True
        else:
            self.smooth_pos = (1.0 - self.lpf_alpha) * self.smooth_pos + self.lpf_alpha * raw_pos
            self.last_valid_quat = raw_quat

        return self.smooth_pos, self.last_valid_quat

    def _check_frame(self, frame_name):
        """Helper to check is an especic frame is available at TF."""
        try:
            trans = self.tf_buffer.lookup_transform("world", frame_name, rospy.Time(0), rospy.Duration(0.01))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            return pos, quat
        except:
            return None

class LandingTraj(BaseTraj):
    """
    Trajectory generator for inclined landing.
    Calculates vectors based on AprilTag normal.
    """
    def __init__(self, tag_observer):
        super().__init__()
        self.tag_obs = tag_observer
        self.is_pressing = False
        self.press_start_time = None
        self.is_touchdown = False

    def get_3d_pt(self, t_cal):
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None: return np.zeros(9) # Hover safety

        t_pos, t_quat = tag_data
        r_mat = R.from_quat(t_quat)
        normal_vec = r_mat.apply([0, 0, 1]) # Normal of the sloped surface
        
        # Get current odom from the observer/buffer logic
        # For simplicity, target distance logic is moved here:
        if not self.is_pressing:
            # Linear approach along normal
            target_dist = max(PRESS_TARGET_DIST, 1.0 - (DESCENT_SPEED * t_cal))
        else:
            target_dist = PRESS_TARGET_DIST

        pos = t_pos + (normal_vec * target_dist)
        vel = -normal_vec * DESCENT_SPEED
        return np.concatenate([pos, vel, np.zeros(3)])

    def get_3d_orientation(self, t_cal):
        tag_data = self.tag_obs.get_smoothed_transform()
        if tag_data is None: return [1,0,0,0,0,0,0,0,0,0]
        _, t_quat = tag_data
        return [t_quat[3], t_quat[0], t_quat[1], t_quat[2], 0,0,0,0,0,0]

    def check_finished(self, t_elapsed):
        return self.is_touchdown

    def get_frame_id(self): return "world"
    def get_child_frame_id(self): return "cog"

if __name__ == "__main__":
    rospy.init_node('beetle_omni_visual_landing')
    robot_name = "beetle1"
    
    # Initialize Vision
    observer = TagObserver()
    
    # Initialize Trajectory Object
    landing_traj = LandingTraj(observer)
    
    # Use MPCTrajPtPub API
    mpc_node = MPCTrajPtPub(robot_name, landing_traj)
    
    # Halt publisher for motor cut
    halt_pub = rospy.Publisher(f"/{robot_name}/teleop_command/halt", Empty, queue_size=1)
    
    rospy.loginfo(f"Landing Mission Started")
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if mpc_node.is_finished:
            rospy.logwarn("Touchdown! Cutting Motors.")
            for _ in range(5): halt_pub.publish(Empty())
            break
        rate.sleep()
