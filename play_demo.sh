#!/bin/bash
# Run INSIDE the container. Plays a demo bag back as if the camera were live, and
# rebuilds the colored point cloud from it (the bag stores color + depth, not the cloud).
#
# Usage:  bash play_demo.sh <bag> [extra rosbag play args]
#   e.g.  bash play_demo.sh /root/kinova_ws/demos/pick_place/pick_place_take01_*.bag
#         bash play_demo.sh <bag> -r 0.5        # half speed
#         bash play_demo.sh <bag> -s 30 -l      # start 30 s in, loop
# Keys while playing: SPACE pause/resume, s = step one message (while paused).
#
# Then, in other terminals:
#   viz               RViz: PointCloud2 /camera/depth_registered/points, Image /camera/rgb/image_raw
#   rqt_image_view    plain image window
# Stop the live camera (start_camera.sh) first - the bag uses the same topic names.

set -eo pipefail
source /opt/ros/noetic/setup.bash
BAG="${1:?Usage: bash play_demo.sh <bag> [rosbag play args]}"; shift
[ -f "$BAG" ] || { echo "No such bag: $BAG"; exit 1; }

if ! timeout 3 rosnode list &>/dev/null; then
  echo "No ROS master running - start one in another terminal with:  roscore"; exit 1
fi
if rosnode list 2>/dev/null | grep -q '^/camera/'; then
  echo "The live camera is running (/camera nodes). Stop start_camera.sh first."; exit 1
fi

# Use the bag's timestamps as "now", so tf and the images line up during playback.
rosparam set use_sim_time true
cleanup() {
  timeout 5 rosnode kill /demo_cloud &>/dev/null || kill "$PC" 2>/dev/null || true
  rosparam set use_sim_time false        # back to normal for live use
  echo "Playback finished; use_sim_time reset to false."
}
trap cleanup EXIT

# Rebuild the point cloud: color + registered depth + intrinsics -> PointCloud2
rosrun nodelet nodelet standalone depth_image_proc/point_cloud_xyzrgb \
  rgb/camera_info:=/camera/rgb/camera_info \
  rgb/image_rect_color:=/camera/rgb/image_raw \
  depth_registered/image_rect:=/camera/depth_registered/image_raw \
  depth_registered/points:=/camera/depth_registered/points \
  __name:=demo_cloud &>/tmp/demo_cloud.log &
PC=$!
sleep 2

echo "Playing $BAG  (SPACE = pause, s = step)"
echo "Point cloud: /camera/depth_registered/points  |  image: /camera/rgb/image_raw"
rosbag play --clock "$BAG" "$@" || true


bash play_demo.sh /root/kinova_ws/demos/pick_place/pick_place_take01_*.bag

bash play_demo.sh /root/kinova_ws/demos/bowl_drop/bowl_drop_take01_*.bag
bash play_demo.sh /root/kinova_ws/demos/slide_path/slide_path_take01_*.bag
bash play_demo.sh /root/kinova_ws/demos/stack2/stack2_take02_*.bag