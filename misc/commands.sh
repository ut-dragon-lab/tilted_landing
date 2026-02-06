# simulation
roslaunch beetle_omni bringup_nmpc_omni.launch real_machine:=false simulation:=True headless:=False nmpc_mode:=0 end_effector:=downward_cam world_type:=1

# remember to rosset..
rossetrobot <robot ip>
rossetip <my ip>

# rosbag, please change it
rosbag record -a -x "(.*)image(.*)|(.*)camera_info(.*)"
