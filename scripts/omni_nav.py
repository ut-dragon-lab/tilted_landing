#!/usr/bin/env python3
"""
Omni Navigation Node for MPC Trajectory Interpolation.
Follows SOLID principles (OCP) and implements safety gatekeeping.
"""

import sys
import argparse
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory

# Import base classes from the provided package structure
from aerial_robot_planning.trajs import BaseTraj
from aerial_robot_planning.pub_mpc_joint_traj import MPCTrajPtPub


class LinearInterpTraj(BaseTraj):
    """
    Trajectory class that interpolates a straight line between two points.
    Open for extension: can be inherited to add orientation smoothing.
    """
    def __init__(self, start_pos, end_pos, speed, loop_num=1):
        super().__init__(loop_num=loop_num)
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.speed = speed
        
        # Calculate direction and total time
        diff = self.end_pos - self.start_pos
        self.distance = np.linalg.norm(diff)
        self.direction = diff / self.distance if self.distance > 0 else np.zeros(3)
        self.T = self.distance / self.speed if self.speed > 0 else 0.0
        
        # Velocity components for feedforward
        self.vel_vec = self.direction * self.speed if self.distance > 0 else np.zeros(3)

    def get_3d_pt(self, t):
        """Returns (x, y, z, vx, vy, vz, ax, ay, az) at time t."""
        if t >= self.T:
            return (*self.end_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        curr_pos = self.start_pos + self.direction * self.speed * t
        return (*curr_pos, *self.vel_vec, 0.0, 0.0, 0.0)

    def check_finished(self, t):
        # The trajectory is finished when time exceeds duration
        return t > (self.T + 0.5)  # 0.5s buffer to ensure MPC settles


class GatekeeperMPCPub(MPCTrajPtPub):
    """
    Publisher that implements a safety gatekeeper.
    Closed for editing: It extends the base publisher functionality.
    """
    MAX_ALLOWED_DISTANCE = 0.2  # meters

    def pub_trajectory_points(self, msg):
        """
        Overrides the publication to validate the trajectory sequence.
        """
        if isinstance(msg, tuple):
            traj_msg, _ = msg
        else:
            traj_msg = msg

        if self._is_sequence_safe(traj_msg):
            super().pub_trajectory_points(msg)
        else:
            rospy.logerr_throttle(2, f"[{self.node_name}] Gatekeeper: Points too far apart! Blocking command.")

    def _is_sequence_safe(self, traj_msg: MultiDOFJointTrajectory) -> bool:
        """
        Validates that no two consecutive points in the horizon exceed 40cm.
        """
        points = traj_msg.points
        for i in range(len(points) - 1):
            p1 = points[i].transforms[0].translation
            p2 = points[i+1].transforms[0].translation
            
            dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
            
            if dist > self.MAX_ALLOWED_DISTANCE:
                return False
        return True


def get_current_pose(robot_name):
    """Fetches the latest odometry to determine the starting point."""
    topic = f"/{robot_name}/uav/cog/odom"
    try:
        rospy.loginfo(f"Waiting for current position on {topic}...")
        odom = rospy.wait_for_message(topic, Odometry, timeout=5.0)
        return odom.pose.pose
    except rospy.ROSException:
        rospy.logerr("Could not acquire robot position. Is the simulation running?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Omni Nav MPC Interpolator")
    parser.add_argument("--robot", type=str, default="beetle1", help="Robot name")
    parser.add_argument("--mode", type=str, required=True, choices=['mission', 'abort'], help="Execution mode")
    parser.add_argument("--target", type=str, default="2.0 2.0 1.2", help="Target x y z (space separated)")
    parser.add_argument("--speed", type=float, default=0.3, help="Speed in m/s")
    
    args = parser.parse_args()
    rospy.init_node("omni_nav_interpolator")

    # 1. Determine Start Position
    start_pose = get_current_pose(args.robot)
    start_xyz = [start_pose.position.x, start_pose.position.y, start_pose.position.z]

    # 2. Determine End Position based on Mode
    if args.mode == 'mission':
        try:
            end_xyz = [float(x) for x in args.target.split()]
        except ValueError:
            rospy.logerr("Invalid target format. Use 'x y z'")
            return
    else: # abort mode
        # Return to XY (0,0) while maintaining current Z
        end_xyz = [0.0, 0.0, start_xyz[2]]
        rospy.logwarn(f"ABORT MODE: Returning to origin at altitude {end_xyz[2]}m")

    # 3. Create Trajectory Object (The "Open" part of OCP)
    traj = LinearInterpTraj(start_xyz, end_xyz, args.speed)

    # 4. Initialize the Safe Publisher
    # This automatically starts the timer at 50Hz and fills the horizon
    pub = GatekeeperMPCPub(args.robot, traj)

    rospy.loginfo(f"Moving from {start_xyz} to {end_xyz} at {args.speed} m/s")
    
    while not rospy.is_shutdown() and not pub.is_finished:
        rospy.sleep(0.1)

    rospy.loginfo("Navigation task completed.")


if __name__ == "__main__":
    main()
