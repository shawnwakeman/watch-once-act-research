#!/bin/bash
# Run INSIDE the container. Builds the ROS workspace (first time takes several minutes).
# Usage:  bash /root/kinova_ws/src/shawn_scripts/build_ws.sh

set -eo pipefail

source /opt/ros/noetic/setup.bash
export CONAN_REVISIONS_ENABLED=1          # Kinova's Conan server needs this
cd /root/kinova_ws

# Gen3 Lite has no Kinova Vision module - skip that package
touch src/ros_kortex_vision/CATKIN_IGNORE

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
