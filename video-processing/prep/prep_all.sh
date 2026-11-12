#!/bin/bash
# Run prep_runs.py on every task bag, with the same three session videos.
# Usage (on the Mac, inside the venv):
#   bash prep_all.sh <bags_folder> <phone_video> <ipad_video> <mac_video> [out_folder]
# Bags folder = the demos/ folder copied from the lab PC (one subfolder per task).
set -eo pipefail
BAGS="${1:?bags folder}"; PHONE="${2:?phone video}"; IPAD="${3:?ipad video}"; MAC="${4:?mac video}"
OUT="${5:-runs}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for bag in "$BAGS"/*/*.bag; do
  case "$(basename "$(dirname "$bag")")" in dev_*) continue ;; esac   # skip practice takes
  echo "================ $bag"
  python3 "$HERE/prep_runs.py" --bag "$bag" \
    --video phone="$PHONE" --video ipad="$IPAD" --video mac="$MAC" --out "$OUT"
done
echo "All tasks done. Check every $OUT/<task>/runXX/check.jpg."
