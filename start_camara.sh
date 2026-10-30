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

# PORT MATTERS: plug the camera into the port that lands on Bus 003 (check on the host
# with `lsusb -t`). On Bus 001 the controller refuses it ("Not enough bandwidth",
# then OpenNI2 says "Failed to set USB interface").
#
# Harmless warnings: "failed to set priority" and "Reconnect has been enabled".
#
# No pre-check here: USB listing tools are unreliable inside Docker. If the driver
# keeps printing "0 devices connected", check on the HOST with: lsusb | grep 1d27

# Camera intrinsics: use the checkerboard calibration saved in the repo if there is one
# (made with cameracalibrator.py, copied to calibration/rgb_PS1080_PrimeSense.yaml).
# Otherwise the driver falls back to generic defaults (a few % off).
INTR="/root/kinova_ws/src/shawn_scripts/calibration/rgb_PS1080_PrimeSense.yaml"
if [ -f "$INTR" ]; then
  echo "Using camera intrinsics from $INTR"
  set -- rgb_camera_info_url:="file://$INTR" "$@"
else
  echo "No intrinsics file in the repo - using driver defaults."
fi

# Only one program can use the camera at a time - close NiViewer2 first.
roslaunch openni2_launch openni2.launch depth_registration:=true "$@"