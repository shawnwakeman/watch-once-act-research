#!/bin/bash
# Run INSIDE the container, with start_driver.sh and start_camera.sh already running.
# Starts ArUco detection + the easy_handeye sampling GUI.
#
# Usage:  bash calibrate.sh <marker_id> <marker_size_m> [dictionary]
#   e.g.  bash calibrate.sh 0 0.100                  # chev.me "4x4" marker (default)
#         bash calibrate.sh 0 0.100 ORIGINAL         # "Original ArUco" marker (aruco_ros)
#         bash calibrate.sh 0 0.100 DICT_6X6_1000    # any other OpenCV dictionary
#
# Check the marker is detected:  rosrun rqt_image_view rqt_image_view /aruco_single/result
# (outline + axes drawn on it = detected)
# When done, click Save in the GUI, then:  bash save_calibration.sh

set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash calibrate.sh <marker_id> <marker_size_m> [dictionary]   e.g. bash calibrate.sh 0 0.100"
  exit 1
fi
ID="$1"; SIZE="$2"; DICT="${3:-DICT_4X4_1000}"

source /opt/ros/noetic/setup.bash
source /root/kinova_ws/devel/setup.bash
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$DICT" != "ORIGINAL" ]; then
  echo "Starting OpenCV ArUco detector ($DICT, id $ID, ${SIZE} m) ..."
  python3 "$DIR/aruco_cv_node.py" _marker_id:="$ID" _marker_size:="$SIZE" _dictionary:="$DICT" &
  DET_PID=$!
  trap 'kill $DET_PID 2>/dev/null' EXIT
fi

roslaunch "$DIR/launch/calibrate.launch" marker_id:="$ID" marker_size:="$SIZE" dictionary:="$DICT"