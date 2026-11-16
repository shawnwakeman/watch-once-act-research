#!/usr/bin/env python3
"""Run perceive.py + lift.py on every run, then collect the results into one table.

Usage (Mac, perception venv, from video-processing/):
  python3 perception/run_all.py                    # all tasks, all runs, L0 (xtion)
  python3 perception/run_all.py --task stack2      # one task
  python3 perception/run_all.py --force            # redo runs that already have results
  python3 perception/run_all.py --only-summary     # just rebuild the table

Each run is processed in its own process (memory is freed between runs). A run that
already has trajectory.yaml is skipped unless --force. Failures are logged and the
batch carries on.

Writes results/<source>_summary.csv: one row per run with detection score, tracking
visibility, Kabsch fit, start/end/displacement, rotation, and end-vs-target offset.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# What to look for in each task. target = a fixed object the trajectory is measured
# against (detected in frame 0 only, not tracked).
TASKS = {
    "pick_place": {"prompt": "pink cube"},
    "bowl_drop":  {"prompt": "pink cube", "target": "cardboard box"},
    "slide_path": {"prompt": "pink cube"},
    "stack2":     {"prompt": "pink cube", "target": "grey block"},
}
HERE = Path(__file__).resolve().parent


def run_one(run_dir, cfg, source, force, log):
    out = run_dir / "perception" / source
    if (out / "trajectory.yaml").exists() and not force:
        return "skipped (done)"
    cmd = [sys.executable, str(HERE / "perceive.py"), str(run_dir), "--prompt", cfg["prompt"], "--source", source]
    if cfg.get("target"):
        cmd += ["--target", cfg["target"]]
    steps = [cmd]
    if source == "xtion":                      # lift.py needs real depth (L0); other levels come later
        steps.append([sys.executable, str(HERE / "lift.py"), str(run_dir)])
    for c in steps:
        r = subprocess.run(c, capture_output=True, text=True)
        log.write(f"\n===== {' '.join(c)}\n{r.stdout}\n{r.stderr}\n")
        log.flush()
        if r.returncode != 0:
            last = (r.stderr.strip() or r.stdout.strip()).splitlines()[-1:] or ["?"]
            return f"FAILED in {Path(c[1]).name}: {last[0][:120]}"
    return "ok"


def summarize(root, source):
    rows = []
    for task in sorted(TASKS):
        for run_dir in sorted((root / task).glob("run*")):
            p = run_dir / "perception" / source
            row = {"task": task, "run": run_dir.name}
            if (p / "detections.json").exists():
                d = json.loads((p / "detections.json").read_text())
                row["dino_score"] = round(d["object"]["score"], 3)
                if "target" in d:
                    row["target_score"] = round(d["target"]["score"], 3)
            if (p / "tracks.npz").exists():
                v = np.load(p / "tracks.npz")["visible"].mean(1)
                row["visible_mean"] = round(float(v.mean()), 3)
                row["visible_min"] = round(float(v.min()), 3)
            if (p / "trajectory.yaml").exists():
                t = yaml.safe_load((p / "trajectory.yaml").read_text())
                dx, dy, dz = t["displacement_m"]
                row.update({
                    "duration_s": round(t["duration_s"], 2),
                    "fit_rms_mm": t["fit_rms_mm_median"], "points_used": t["points_used_median"],
                    "frames_held": t["frames_held_too_few_points"],
                    "start_x": t["start_base_m"][0], "start_y": t["start_base_m"][1], "start_z": t["start_base_m"][2],
                    "end_x": t["end_base_m"][0], "end_y": t["end_base_m"][1], "end_z": t["end_base_m"][2],
                    "disp_xy_cm": round(100 * float(np.hypot(dx, dy)), 1), "disp_z_cm": round(100 * dz, 1),
                    "path_cm": round(100 * t["path_length_m"], 1), "end_rot_deg": t["end_rotation_deg"],
                })
                if "end_minus_target_m" in t:
                    ex, ey, ez = t["end_minus_target_m"]
                    row.update({"end_vs_target_xy_cm": round(100 * float(np.hypot(ex, ey)), 1),
                                "end_vs_target_z_cm": round(100 * ez, 1)})
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--task", choices=sorted(TASKS))
    ap.add_argument("--source", default="xtion")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-summary", action="store_true")
    a = ap.parse_args()

    root = Path(a.runs)
    results = Path("results")
    results.mkdir(exist_ok=True)
    if not a.only_summary:
        with open(results / f"{a.source}_log.txt", "a") as log:
            for task in ([a.task] if a.task else sorted(TASKS)):
                for run_dir in sorted((root / task).glob("run*")):
                    print(f"{task}/{run_dir.name} ... ", end="", flush=True)
                    print(run_one(run_dir, TASKS[task], a.source, a.force, log))
        print(f"(full output of every step: results/{a.source}_log.txt)")

    rows = summarize(root, a.source)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = results / f"{a.source}_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary: {path}")
    print(f"{'task/run':22s} {'dino':>5s} {'vis':>5s} {'fit mm':>6s} {'pts':>4s} {'disp cm':>7s} {'rot':>5s} {'vs target cm':>12s}")
    for r in rows:
        print(f"{r['task'] + '/' + r['run']:22s} {r.get('dino_score', '-')!s:>5s} {r.get('visible_mean', '-')!s:>5s} "
              f"{r.get('fit_rms_mm', '-')!s:>6s} {r.get('points_used', '-')!s:>4s} {r.get('disp_xy_cm', '-')!s:>7s} "
              f"{r.get('end_rot_deg', '-')!s:>5s} {r.get('end_vs_target_xy_cm', '-')!s:>12s}")


if __name__ == "__main__":
    main()