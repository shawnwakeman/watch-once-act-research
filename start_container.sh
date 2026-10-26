#!/bin/bash
# Opens a shell in personal container (run on the lab PC, not inside Docker).
#   - first run:            clones the repo into ~/shawn and creates the container
#   - container stopped:    restarts it
#   - container running:    opens a second terminal into it
#
# The container gets: the PC's network (to reach the arm), USB devices (depth camera),
# and the PC's display (RViz windows).
#
# Usage:  bash ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/start_container.sh

set -e

REPO="$HOME/shawn/kinova-gen3-docker-ros"
NAME="kinova-shawn"
WORKDIR="/root/kinova_ws/src/shawn_scripts"   # where new shells start
IMAGE="kinova-gen3-dev:latest"

if [ ! -d "$REPO" ]; then
  echo "Cloning workspace into $REPO ..."
  mkdir -p "$HOME/shawn"
  git clone https://github.com/umn-cse-mnri/kinova-gen3-docker-ros "$REPO"
fi

# Let the container open windows on this screen (RViz)
xhost +local: > /dev/null 2>&1 || true

# Bump CFG whenever the docker run flags below change; older containers get replaced.
CFG=5
if [ -n "$(docker ps -aq -f name=^${NAME}$)" ] && \
   [ "$(docker inspect -f '{{index .Config.Labels "shawn.cfg"}}' "$NAME")" != "$CFG" ]; then
  echo "Existing $NAME was made with older settings - recreating it (your files are safe)."
  docker rm -f "$NAME" > /dev/null
  RECREATED=1
fi

if [ -n "$(docker ps -q -f name=^${NAME}$)" ]; then
  echo "Container $NAME is running - opening another terminal into it."
  docker exec -it -w "$WORKDIR" "$NAME" bash
elif [ -n "$(docker ps -aq -f name=^${NAME}$)" ]; then
  echo "Restarting existing container $NAME."
  docker start -ai "$NAME"
else
  echo "Creating container $NAME (your files live in $REPO/kinova_ws)."
  [ -n "$RECREATED" ] && echo ">> Run build_ws.sh inside it once to reinstall packages and terminal setup."
  # /run/udev: USB drivers (OpenNI2/libusb) find devices through udev, which a
  # container doesn't run - sharing the host's udev info lets the camera be found.
  # /sys: the USB device list libusb reads; rootless Docker doesn't provide it.
  # (Don't mount all of /dev: it breaks the container's terminal - "reopen ptmx" error.)
  docker run -it --name "$NAME" -w "$WORKDIR" --label shawn.cfg=$CFG --net=host --privileged \
    -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/bus/usb:/dev/bus/usb -v /run/udev:/run/udev:ro -v /sys:/sys:ro \
    -v "$REPO/kinova_ws":/root/kinova_ws \
    "$IMAGE"
fi