#!/bin/bash
# Opens a shell in personal container,
#   - first run:            clones the repo into ~/shawn and creates the container
#   - container stopped:    restarts it
#   - container running:    opens a second terminal into it
#
# Usage:  bash ~/shawn/kinova-gen3-docker-ros/kinova_ws/src/shawn_scripts/start_container.sh

set -e

REPO="$HOME/shawn/kinova-gen3-docker-ros"
NAME="kinova-shawn"
IMAGE="kinova-gen3-dev:latest"

if [ ! -d "$REPO" ]; then
  echo "Cloning workspace into $REPO ..."
  mkdir -p "$HOME/shawn"
  git clone https://github.com/umn-cse-mnri/kinova-gen3-docker-ros "$REPO"
fi

if [ -n "$(docker ps -q -f name=^${NAME}$)" ]; then
  echo "Container $NAME is running - opening another terminal into it."
  docker exec -it "$NAME" bash
elif [ -n "$(docker ps -aq -f name=^${NAME}$)" ]; then
  echo "Restarting existing container $NAME."
  docker start -ai "$NAME"
else
  echo "Creating container $NAME (your files live in $REPO/kinova_ws)."
  docker run -it --name "$NAME" --net=host \
    -v "$REPO/kinova_ws":/root/kinova_ws \
    "$IMAGE"
fi
