#!/bin/bash
# Run INSIDE the container with start_camera.sh running (and publish_calibration.py,
# so the camera's pose in base_link is saved with the demo).
# Records one demo take as a rosbag: color + registered depth + intrinsics + tf_static.
#
# Usage:  bash record_demo.sh <task_name> [seconds]
#   e.g.  bash record_demo.sh pick_place           # records until you type q + Enter
#         bash record_demo.sh dev_dryrun 10        # stops by itself after 10 s
#
# Several runs in one recording: press ENTER to mark each run's start and end.
#   slate + slap -> Enter (run 1 START) -> do the task -> Enter (run 1 END) -> reset ->
#   slate + slap -> Enter (run 2 START) -> ...   then  q + Enter  to stop recording.
# The marks are saved next to the bag as <bag name>.markers.csv (event,run,unix_time),
# which is what the cutting script uses to split the bag and the other videos into runs.
# Takes are numbered automatically:  /root/kinova_ws/demos/<task>/<task>_take03_<date>.bag
# Bags live OUTSIDE the git repo (they're ~40 MB per second) but on the mounted folder,
# so they survive the container. Export frames with:  python3 extract_demo.py <bag>

set -eo pipefail
source /opt/ros/noetic/setup.bash
TASK="${1:?Usage: bash record_demo.sh <task_name> [seconds]}"
SECS="${2:-}"
if [ -n "$SECS" ] && ! [[ "$SECS" =~ ^[0-9]+$ ]]; then
  echo "Second argument must be a number of seconds (got '$SECS')."
  echo "Task names can't contain spaces - use e.g.  bash record_demo.sh pick_place"
  exit 1
fi
echo "Checking the camera and calibration (a few seconds)..."
DIR="/root/kinova_ws/demos/$TASK"
mkdir -p "$DIR"

TOPICS=(/camera/rgb/image_raw /camera/rgb/camera_info
        /camera/depth_registered/image_raw /camera/depth_registered/camera_info
        /tf_static)

# --- checks -------------------------------------------------------------------------
for t in /camera/rgb/image_raw /camera/depth_registered/image_raw; do
  if ! timeout 5 rostopic echo -n1 "$t/header" &>/dev/null; then
    echo "No messages on $t - is start_camera.sh running (and nothing else using the camera)?"
    exit 1
  fi
done
TF_OUT=$(timeout 4 rosrun tf tf_echo base_link camera_link 2>/dev/null || true)
grep -q Translation <<< "$TF_OUT" \
  || echo "WARNING: no base_link -> camera_link (publish_calibration.py not running?). Recording anyway."
FREE_GB=$(df -BG --output=avail /root/kinova_ws | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -lt 5 ] && echo "WARNING: only ${FREE_GB} GB free on the workspace disk."

N=$(( $(find "$DIR" -maxdepth 1 -name "*.bag" | wc -l) + 1 ))
NAME="${TASK}_take$(printf %02d $N)_$(date +%Y-%m-%d_%H%M%S)"
echo
echo "Recording take $N of '$TASK' -> $DIR/$NAME.bag"
[ -n "$SECS" ] && echo "(stops automatically after $SECS s)"
for i in 3 2 1; do echo "  $i..."; sleep 1; done

ARGS=(-O "$DIR/$NAME.bag" --lz4 "${TOPICS[@]}")
[ -n "$SECS" ] && ARGS=(--duration="$SECS" "${ARGS[@]}")
MARK="$DIR/$NAME.markers.csv"
echo "event,run,unix_time" > "$MARK"

# rosbag runs in the background (its own process group, so a stray Ctrl+C doesn't hit
# it directly); this script reads Enter presses as run markers, then stops rosbag with
# SIGINT so it closes the bag cleanly.
set -m
rosbag record -q "${ARGS[@]}" &
RB=$!
set +m
stop_rec() { kill -INT "$RB" 2>/dev/null || true; }
trap 'echo; echo "  stopping..."; stop_rec' INT
sleep 1 || true
read -r -t 0.1 -n 10000 _DRAIN || true     # ignore Enter presses made before this point
echo "  RECORDING."
echo
echo "  Slate + slap, then ENTER = run 1 START.     (q + Enter = stop recording)"
RUN=1; STATE=start
while kill -0 "$RB" 2>/dev/null; do
  KEY=""
  read -r -t 1 KEY || { [ $? -gt 128 ] && continue; break; }   # timeout -> keep waiting
  [ "$KEY" = q ] && break
  T=$(date +%s.%N)
  echo "$STATE,$RUN,$T" >> "$MARK"
  if [ "$STATE" = start ]; then
    echo "  >> run $RUN STARTED  - do the task, then ENTER = run $RUN END"
    STATE=end
  else
    echo "  >> run $RUN ENDED    - reset, slate + slap, then ENTER = run $((RUN + 1)) START  (q + Enter = stop)"
    RUN=$((RUN + 1)); STATE=start
  fi
done
stop_rec
wait "$RB" 2>/dev/null || true
trap - INT
[ "$STATE" = end ] && echo "  (run $RUN was started but not ended - it'll be cut at the end of the recording)"
echo "  Marked $(( $(grep -c '^end,' "$MARK" || true) )) complete run(s) -> $MARK"

echo
rosbag info "$DIR/$NAME.bag" | grep -E "duration|size|messages|image_raw" || true
read -r -p "Note for this take (what happened / keep or bad), Enter to skip: " NOTE
echo "$(date +%F\ %T),$NAME,${NOTE//,/;}" >> "$DIR/takes.csv"
echo "Logged to $DIR/takes.csv"