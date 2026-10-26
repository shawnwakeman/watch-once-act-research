#!/bin/bash
# Run INSIDE the container on your Mac, once (again only if the container is recreated).
# Installs what the simulation needs, then builds the workspace.
#   - ros-noetic-ros-controllers: the joint controllers Gazebo was missing
#     ("controller type 'effort_controllers/...' does not exist")
#   - xvfb/x11vnc/fluxbox/novnc/websockify: a desktop you view in your browser
# Usage:  bash /root/kinova_ws/src/shawn_scripts/setup_sim.sh

set -eo pipefail

apt-get update
apt-get install -y ros-noetic-ros-controllers xvfb x11vnc fluxbox novnc websockify

bash "$(dirname "$0")/build_ws.sh"
echo
echo "Sim setup done. Next:  bash /root/kinova_ws/src/shawn_scripts/start_sim.sh"
