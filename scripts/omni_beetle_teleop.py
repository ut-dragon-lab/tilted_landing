#!/usr/bin/env python3
import rospy
import sys, select, termios, tty
from geometry_msgs.msg import Pose, Quaternion, Vector3
from nav_msgs.msg import Odometry
from aerial_robot_planning.pub_mpc_joint_traj import MPCSinglePtPub

# Key mappings
msg = """
Beetle NMPC Teleop Navigation
---------------------------
W / S : North (+) / South (-) [X]
A / D : West  (+) / East  (-) [Y]
[ / ] : Up    (+) / Down  (-) [Z]

CTRL-C to quit
"""

def get_key():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('beetle_teleop_nav')
    
    robot_name = "beetle1"
    step = 0.1 # Meters per key press
    
    # Initialize target from current odometry
    print("Waiting for odom...")
    odom = rospy.wait_for_message(f"/{robot_name}/uav/cog/odom", Odometry)
    curr_x = odom.pose.pose.position.x
    curr_y = odom.pose.pose.position.y
    curr_z = odom.pose.pose.position.z
    curr_q = odom.pose.pose.orientation

    print(msg)
    
    try:
        while not rospy.is_shutdown():
            key = get_key()
            if key == 'w': curr_x += step
            elif key == 's': curr_x -= step
            elif key == 'a': curr_y += step
            elif key == 'd': curr_y -= step
            elif key == '[': curr_z += step
            elif key == ']': curr_z -= step
            elif key == '\x03': break # Ctrl-C
            else: continue

            # Update NMPC target
            target_pose = Pose(position=Vector3(curr_x, curr_y, curr_z), orientation=curr_q)
            # Instantiate SinglePtPub to handle the smooth transition to the new waypoint
            mpc_node = MPCSinglePtPub(robot_name, "world", "cog", target_pose)
            
            rospy.loginfo(f"Target Updated: X:{curr_x:.2f} Y:{curr_y:.2f} Z:{curr_z:.2f}")

    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
