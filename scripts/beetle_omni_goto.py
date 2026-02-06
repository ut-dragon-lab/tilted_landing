#!/usr/bin/env python3
import rospy
import sys
import tf_conversions as tf
from geometry_msgs.msg import Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from aerial_robot_planning.pub_mpc_joint_traj import MPCSinglePtPub

def print_usage():
    print("Usage:")
    print("  rosrun tilted_landing omni_beetle_goto.py <x> <y> <z>")
    print("  rosrun tilted_landing omni_beetle_goto.py <x> <y> <z> <yaw>")
    print("  rosrun tilted_landing omni_beetle_goto.py <x> <y> <z> <roll> <pitch> <yaw>")

def main():
    # 1. Parse command line arguments
    if len(sys.argv) < 4:
        print_usage()
        return

    try:
        target_x = float(sys.argv[1])
        target_y = float(sys.argv[2])
        target_z = float(sys.argv[3])
    except ValueError:
        print("Error: Coordinates must be numeric.")
        return

    # 2. Initialize ROS Node
    rospy.init_node('beetle_goto_cli')
    robot_name = "beetle1"
    
    # 3. Get current orientation as default
    rospy.loginfo("Fetching current robot state...")
    try:
        # We wait for the first odom message to ensure we have a valid orientation
        odom = rospy.wait_for_message(f"/{robot_name}/uav/cog/odom", Odometry, timeout=5.0)
        target_q = odom.pose.pose.orientation
    except rospy.ROSException:
        rospy.logerr("Timeout waiting for odometry. Ensure the robot is running.")
        return

    # 4. Optional Attitude Parsing (Euler Radians -> Quaternion)
    if len(sys.argv) == 5: # Handle Yaw only
        yaw = float(sys.argv[4])
        q_raw = tf.transformations.quaternion_from_euler(0, 0, yaw)
        target_q = Quaternion(*q_raw)
        rospy.loginfo(f"Targeting Yaw: {yaw:.2f} rad")
        
    elif len(sys.argv) == 7: # Handle Roll, Pitch, Yaw
        r = float(sys.argv[4])
        p = float(sys.argv[5])
        y = float(sys.argv[6])
        q_raw = tf.transformations.quaternion_from_euler(r, p, y)
        target_q = Quaternion(*q_raw)
        rospy.loginfo(f"Targeting Attitude: R:{r:.2f}, P:{p:.2f}, Y:{y:.2f}")

    # 5. Execute Command via MPCSinglePtPub
    # This class handles the smooth transition to the setpoint using NMPC
    target_pose = Pose(position=Vector3(target_x, target_y, target_z), orientation=target_q)
    rospy.loginfo(f"Sending Beetle to: X:{target_x}, Y:{target_y}, Z:{target_z}")
    
    mpc_node = MPCSinglePtPub(robot_name, "world", "cog", target_pose)
    
    # 6. Monitor Progress
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # check_finished() in MPCSinglePtPub returns True when within tolerances
        if mpc_node.is_finished:
            rospy.loginfo("Target Reached successfully.")
            break
        rate.sleep()

if __name__ == "__main__":
    main()
