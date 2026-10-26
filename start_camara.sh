#!/bin/bash
# Run INSIDE the container. Starts the depth camera (ASUS Xtion / PrimeSense, 1d27:0600)
# with depth aligned to color. Leave this terminal running.
# Usage:  bash /root/kinova_ws/src/shawn_scripts/start_camera.sh
#
# Check from another terminal:
#   rostopic hz /camera/rgb/image_raw
#   rostopic hz /camera/depth_registered/image_raw
#   rostopic echo -n1 /camera/rgb/camera_info

source /opt/ros/noetic/setup.bash

# No pre-check here: USB listing tools are unreliable inside Docker. If the driver
# keeps printing "0 devices connected", check on the HOST with: lsusb | grep 1d27

roslaunch openni2_launch openni2.launch depth_registration:=true "$@"