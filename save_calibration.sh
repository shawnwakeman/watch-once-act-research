#!/bin/bash
# Run INSIDE the container after clicking "Save" in the calibration GUI.
# Copies the newest eye-on-base result into the repo (calibration/) with a date stamp,
# and as calibration/current.yaml, which publish_calibration.py loads by default.
# (~/.ros lives inside the container and is lost if the container is recreated.)

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$(ls -t "$HOME"/.ros/easy_handeye/*eye_on_base*.yaml 2>/dev/null | head -1)"

if [ -z "$SRC" ]; then
  echo "No calibration file in ~/.ros/easy_handeye/ - did you click Save in the GUI?"
  ls -la "$HOME/.ros/easy_handeye/" 2>/dev/null || true
  exit 1
fi

mkdir -p "$DIR/calibration"
STAMP="$(date +%Y-%m-%d_%H%M)"
cp "$SRC" "$DIR/calibration/handeye_$STAMP.yaml"
cp "$SRC" "$DIR/calibration/current.yaml"
echo "Copied $SRC"
echo " -> calibration/handeye_$STAMP.yaml and calibration/current.yaml"
echo "Commit and push it from the host so it isn't lost."
echo
cat "$DIR/calibration/current.yaml"
