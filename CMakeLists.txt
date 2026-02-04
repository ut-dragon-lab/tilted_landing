cmake_minimum_required(VERSION 3.0.2)
project(tilted_landing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
  smach_ros
  tf2_ros
  std_msgs
  geometry_msgs
  nav_msgs
  trajectory_msgs
  aerial_robot_msgs
  aerial_robot_planning
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    smach 
    smach_ros 
    aerial_robot_msgs 
    aerial_robot_planning
)

catkin_install_python(PROGRAMS
  scripts/landing_smash.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
