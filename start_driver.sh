#!/bin/bash
# Run INSIDE the container. Connects ROS to the real arm and starts MoveIt.
# Leave this terminal running; run your scripts from a second terminal.
# Usage:  bash /root/kinova_ws/src/shawn_scripts/start_driver.sh
# Different arm IP:  ROBOT_IP=192.168.1.12 bash start_driver.sh

source /opt/ros/noetic/setup.bash
source /root/kinova_ws/devel/setup.bash

ROBOT_IP="${ROBOT_IP:-192.168.1.10}"

echo "Checking connection to the arm at $ROBOT_IP ..."
# Kortex API listens on TCP 10000. (No ping here: the container may not have it installed.)
if ! timeout 3 bash -c "</dev/tcp/$ROBOT_IP/10000" 2>/dev/null; then
  echo "Can't reach the arm's API at $ROBOT_IP:10000. Is the USB cable in this PC and the base LED green?"
  exit 1
fi
echo "Arm reachable."

roslaunch kortex_driver kortex_driver.launch \
  arm:=gen3_lite gripper:=gen3_lite_2f \
  ip_address:="$ROBOT_IP" start_rviz:=false