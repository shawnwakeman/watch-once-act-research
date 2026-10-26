#!/bin/bash
# Run INSIDE the container. Builds the ROS workspace (first time takes several minutes).
# Usage:  bash /root/kinova_ws/src/shawn_scripts/build_ws.sh

set -eo pipefail

source /opt/ros/noetic/setup.bash
export CONAN_REVISIONS_ENABLED=1          # Kinova's Conan server needs this
cd /root/kinova_ws

# Gen3 Lite has no Kinova Vision module - skip that package
touch src/ros_kortex_vision/CATKIN_IGNORE

# On an ARM machine (your Mac), Kinova publishes no build of its arm library,
# so skip the real-arm driver. Simulation doesn't need it.
if [ "$(uname -m)" = "aarch64" ]; then
  echo "ARM machine detected: skipping kortex_driver/kortex_examples (simulation only)."
  touch src/ros_kortex/kortex_driver/CATKIN_IGNORE src/ros_kortex/kortex_examples/CATKIN_IGNORE
fi

# Depth camera driver (ASUS Xtion / PrimeSense, USB ID 1d27:0600). Installed in the
# container, so a recreated container needs this again - hence it lives here.
if ! dpkg -s ros-noetic-openni2-launch > /dev/null 2>&1; then
  echo "Installing depth camera driver (openni2) ..."
  apt-get update && apt-get install -y ros-noetic-openni2-launch ros-noetic-rgbd-launch
fi

# Hand-eye calibration tools: ArUco marker detection (apt) + easy_handeye (source)
if ! dpkg -s ros-noetic-aruco-ros > /dev/null 2>&1; then
  echo "Installing ArUco detection (aruco_ros) ..."
  apt-get update && apt-get install -y ros-noetic-aruco-ros
fi
if [ ! -d src/easy_handeye ]; then
  echo "Cloning easy_handeye ..."
  git clone https://github.com/IFL-CAMP/easy_handeye src/easy_handeye
fi
# The image's pip setuptools needs a newer importlib_metadata than it ships with;
# without this, catkin fails on Python packages ("no attribute 'EntryPoints'").
pip3 install -q -U "importlib_metadata>=4.6" --root-user-action=ignore 2>/dev/null || \
  pip3 install -q -U "importlib_metadata>=4.6"

# Install anything easy_handeye needs that isn't here yet (no-op once satisfied)
rosdep install -iy --from-paths src/easy_handeye || \
  echo "rosdep had trouble - if the build fails on easy_handeye, paste the error."

# Make every new terminal in this container ROS-ready (added once)
if ! grep -q "shawn_scripts env" ~/.bashrc; then
  cat >> ~/.bashrc << 'EOF'
# --- shawn_scripts env ---
source /opt/ros/noetic/setup.bash
[ -f /root/kinova_ws/devel/setup.bash ] && source /root/kinova_ws/devel/setup.bash
export CONAN_REVISIONS_ENABLED=1
EOF
fi

catkin_make 2>&1 | tee build.log
echo
echo "Build finished. Run:  source /root/kinova_ws/devel/setup.bash"