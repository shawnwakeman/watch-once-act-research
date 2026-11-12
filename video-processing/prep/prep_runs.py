#!/usr/bin/env python3
"""Cut one task's recording into runs, with all four cameras synced. Runs on the Mac
(no ROS needed - reads bags with the `rosbags` package).

Inputs:
  - the task's Xtion bag and its .markers.csv (from record_demo.sh, Enter presses)
  - the session-long videos from the other cameras (phone, iPad, Mac)

For each run (start Enter -> end Enter) it writes:
  runs/<task>/run01/xtion/rgb/000000.png ...   color frames (L0 reference)
  runs/<task>/run01/xtion/depth/000000.png ... registered depth, 16-bit mm, matched by time
  runs/<task>/run01/xtion/frames.csv            index, rgb_time, depth_time
  runs/<task>/run01/xtion/camera.yaml           K, D, size, base_link->camera_link
  runs/<task>/run01/<camera>.mp4                each video cut to the same span
  runs/<task>/run01/sync.yaml                   offsets + match scores
  runs/<task>/run01/check.jpg                   the clap moment from every camera

How the sync works: each video file has one offset from the lab PC's clock
(pc_time = video_time + offset).
  1. Audio (preferred): the claps are sharp spikes in each video's sound. Their pattern
     (irregular gaps between runs) is matched to the clap moments before each run in the
     Xtion bag - only one offset fits all of them, and camera shake doesn't matter.
  2. The bag has no sound, so the bag<->video link is then fine-tuned (+-0.6 s) by
     matching motion with the video that moves most like the Xtion (usually the iPad).
  3. Fallback if a video has no usable audio: match motion over the span of the runs.
Check every run's check.jpg: all views should show the same instant of the clap.
If one is wrong, pass its offset by hand with --offset (sync.yaml shows the found ones).

Usage:
  python3 prep_runs.py --bag bags/pick_place_take01_....bag \\
      --video phone=raw/IMG_1234.MOV --video ipad=raw/IMG_0456.MOV --video mac=raw/mac.mov \\
      --out runs
  # optional: --offset phone=1794421000.25   (pc_time - video_time, seconds) to force one
  #           --fps 30   output frame rate for the video cuts
"""
import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

RGB = "/camera/rgb/image_raw"
DEPTH = "/camera/depth_registered/image_raw"
INFO = "/camera/rgb/camera_info"
GRID_HZ = 30.0                 # motion curves are compared on a 30 Hz grid
SMALL = (160, 120)             # motion is computed on tiny grayscale frames


# --------------------------------------------------------------------- helpers ---
def motion_energy(frames_gray):
    """Mean absolute change between consecutive tiny frames (first = 0)."""
    e = [0.0]
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        e.append(float(np.mean(cv2.absdiff(a, b))))
    return np.array(e)


def to_grid(t, v, t0, t1, hz=GRID_HZ):
    grid = np.arange(t0, t1, 1.0 / hz)
    return grid, np.interp(grid, t, v)


def zscore(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0


def lag_scores(template, signal, lo, hi):
    """Normalized cross-correlation of template over signal[lag:lag+m] for every lag in
    [lo, hi]. Returns (lags, scores)."""
    m = len(template)
    lo, hi = max(0, lo), min(len(signal) - m, hi)
    if hi < lo or m < 2:
        return np.array([], int), np.array([])
    t = zscore(template)
    c1 = np.concatenate([[0], np.cumsum(signal)])          # sliding mean / std
    c2 = np.concatenate([[0], np.cumsum(signal ** 2)])
    lags = np.arange(lo, hi + 1)
    mu = (c1[lags + m] - c1[lags]) / m
    var = np.maximum((c2[lags + m] - c2[lags]) / m - mu ** 2, 1e-12)
    n = 1 << int(np.ceil(np.log2(len(signal) + m)))       # dot products via FFT
    dots = np.fft.irfft(np.fft.rfft(signal, n) * np.conj(np.fft.rfft(t, n)), n)[lags]
    return lags, dots / (m * np.sqrt(var))


def best_lag(template, signal, lo, hi):
    lags, sc = lag_scores(template, signal, lo, hi)
    if len(sc) == 0:
        return None, -1.0
    i = int(np.argmax(sc))
    return int(lags[i]), float(sc[i])


def rolling_median(x, win):
    """Median of a centered window (edges padded)."""
    from numpy.lib.stride_tricks import sliding_window_view
    h = win // 2
    xp = np.pad(x, (h, win - 1 - h), mode="edge")
    return np.median(sliding_window_view(xp, win), axis=1)


def bag_claps(grid_t, grid_e, runs):
    """The clap before each run in the Xtion bag: the sharpest motion spike (motion minus
    its 1 s running median, so slow movements like raising the slate count less) in the
    6 s before the run's start Enter."""
    spike = grid_e - rolling_median(grid_e, int(GRID_HZ))
    out = {}
    for n, t_start, _ in runs:
        w0, w1 = np.searchsorted(grid_t, [t_start - 6.0, t_start - 0.2])
        if w1 > w0:
            out[n] = float(grid_t[w0 + int(np.argmax(spike[w0:w1]))])
    return out


def audio_onsets(path, cache_dir, sr=16000, hop=80):
    """Sharp sound onsets (claps) in a video's audio: (times_s, strengths). Cached."""
    cache = cache_dir / (Path(path).name + ".onsets.npy")
    if cache.exists() and cache.stat().st_mtime > os.path.getmtime(path):
        d = np.load(cache)
        return d[0], d[1]
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(p.stdout, np.int16).astype(np.float32)
    if len(x) < sr:
        return np.array([]), np.array([])
    hp = np.diff(x)                                    # emphasize sharp transients
    nfr = len(hp) // hop
    env = np.sqrt(np.mean(hp[:nfr * hop].reshape(nfr, hop) ** 2, axis=1)) + 1e-6   # 5 ms frames
    base = rolling_median(env, int(0.5 * sr / hop))                                # 0.5 s background
    ratio = env / (base + 1e-6)
    times, strengths, i, gap = [], [], 0, int(0.5 * sr / hop)
    cand = np.where(ratio > 6.0)[0]
    while i < len(cand):                               # one onset per 0.5 s burst
        j = i
        while j + 1 < len(cand) and cand[j + 1] - cand[i] < gap:
            j += 1
        k = cand[i] + int(np.argmax(ratio[cand[i]:cand[j] + 1]))
        times.append(cand[i] * hop / sr)               # rising edge = the clap instant
        strengths.append(float(ratio[k]))
        i = j + 1
    t, st = np.array(times), np.array(strengths)
    np.save(cache, np.vstack([t, st]) if len(t) else np.zeros((2, 0)))
    return t, st


def match_claps(claps, onsets, windows=None, tol=0.15):
    """Find the offset (pc_time = video_time + offset) that lines up the most bag claps with
    audio onsets. The gaps between runs are irregular, so only one offset fits them all.
    Returns (offset, n_matched, spread_s, {run: offset}) or None."""
    if len(onsets) == 0 or not claps:
        return None
    items = sorted(claps.items())
    best = None
    for _, tb in items:
        for ta in onsets:
            O = tb - ta
            if windows and not any(lo <= O <= hi for lo, hi in windows):
                continue
            per, resid = {}, 0.0
            for n, tb2 in items:
                k = int(np.argmin(np.abs(onsets - (tb2 - O))))
                d = abs(onsets[k] - (tb2 - O))
                if d <= tol:
                    per[n], resid = tb2 - onsets[k], resid + d
            key = (len(per), -resid)
            if best is None or key > best[0]:
                best = (key, per)
    if best is None or len(best[1]) < max(2, int(np.ceil(0.6 * len(items)))):
        return None
    per = best[1]
    vals = np.array(list(per.values()))
    off = float(np.median(vals))
    return off, len(per), float(np.max(np.abs(vals - off))), per


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                          "-show_streams", str(path)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def has_encoder(name):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return name in out.stdout


def creation_epoch(info):
    """Recording start time from the file's metadata, if present (coarse, ~seconds)."""
    tags = dict(info.get("format", {}).get("tags", {}))
    for s in info.get("streams", []):
        tags.update(s.get("tags", {}))
    for k in ("com.apple.quicktime.creationdate", "creation_time"):
        v = tags.get(k)
        if v:
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return None


# ----------------------------------------------------------------- the Xtion bag ---
def _img(msg, topic):
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if topic == RGB:
        img = arr.reshape(msg.height, msg.width, -1)
        return np.ascontiguousarray(img[:, :, ::-1] if msg.encoding.lower() == "rgb8" else img)
    if msg.encoding.upper() == "16UC1":
        return arr.view(np.uint16).reshape(msg.height, msg.width)
    return np.nan_to_num(arr.view(np.float32).reshape(msg.height, msg.width) * 1000).astype(np.uint16)


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def scan_bag(bag_path):
    """Pass 1: timestamps, a tiny motion curve, intrinsics and calibration. Keeps no full
    frames in memory (a 4-minute bag is ~10 GB of images)."""
    from rosbags.highlevel import AnyReader
    rgb_t, depth_t, motion, info, calib, prev = [], [], [], None, None, None
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in (RGB, DEPTH, INFO, "/tf_static")]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == RGB:
                small = cv2.resize(cv2.cvtColor(_img(msg, RGB), cv2.COLOR_BGR2GRAY), SMALL)
                rgb_t.append(_stamp(msg))
                motion.append(0.0 if prev is None else float(np.mean(cv2.absdiff(small, prev))))
                prev = small
            elif conn.topic == DEPTH:
                depth_t.append(_stamp(msg))
            elif conn.topic == INFO and info is None:
                f = lambda name: getattr(msg, name) if hasattr(msg, name) else getattr(msg, name.lower())
                info = {"width": int(msg.width), "height": int(msg.height),   # ROS1 names are K/D/P
                        "K": [float(v) for v in f("K")], "D": [float(v) for v in f("D")],
                        "P": [float(v) for v in f("P")], "distortion_model": msg.distortion_model}
            elif conn.topic == "/tf_static":
                for tr in msg.transforms:
                    if tr.header.frame_id == "base_link" and tr.child_frame_id == "camera_link":
                        t, r = tr.transform.translation, tr.transform.rotation
                        calib = {"x": t.x, "y": t.y, "z": t.z, "qx": r.x, "qy": r.y, "qz": r.z, "qw": r.w}
    order = np.argsort(rgb_t)
    return (np.array(rgb_t)[order], np.array(motion)[order], np.sort(np.array(depth_t)), info, calib)


def write_frames(bag_path, plan):
    """Pass 2: write each run's color frames, and the depth frame nearest each one.
    plan = {run_dir: (t_start, t_end, {depth_time: [frame indices]})} built from pass 1."""
    from rosbags.highlevel import AnyReader
    counters = {d: 0 for d in plan}
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic in (RGB, DEPTH)]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ts = _stamp(msg)
            for rdir, (t0, t1, rgb_index, depth_map) in plan.items():
                if conn.topic == RGB and t0 <= ts <= t1:
                    k = rgb_index.get(round(ts, 6))
                    if k is not None:
                        cv2.imwrite(str(rdir / "xtion" / "rgb" / f"{k:06d}.png"), _img(msg, RGB))
                        counters[rdir] += 1
                elif conn.topic == DEPTH:
                    for k in depth_map.get(round(ts, 6), []):
                        cv2.imwrite(str(rdir / "xtion" / "depth" / f"{k:06d}.png"), _img(msg, DEPTH))
    return counters


def bag_frames_at(bag_path, wanted, rgb_t):
    """Fetch the color frame nearest each wanted time ({key: pc_time}) - for the check sheets."""
    from rosbags.highlevel import AnyReader
    targets = {k: round(float(rgb_t[int(np.argmin(np.abs(rgb_t - t)))]), 6) for k, t in wanted.items()}
    need = set(targets.values()); got = {}
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == RGB]
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            ts = round(_stamp(msg), 6)
            if ts in need:
                got[ts] = _img(msg, RGB)
    return {k: got[v] for k, v in targets.items() if v in got}


def read_markers(path):
    starts, ends = {}, {}
    for line in open(path).read().strip().splitlines()[1:]:
        ev, n, t = line.split(",")
        (starts if ev == "start" else ends)[int(n)] = float(t)
    runs = []
    for n in sorted(starts):
        runs.append((n, starts[n], ends.get(n)))  # end None -> until the bag ends
    return runs


# --------------------------------------------------------------------- videos ---
def video_motion(path, cache_dir):
    """Motion curve of a whole video at 30 Hz (cached - the long videos are shared by all tasks)."""
    cache = cache_dir / (Path(path).name + ".motion.npy")
    if cache.exists() and cache.stat().st_mtime > os.path.getmtime(path):
        return np.load(cache)
    w, h = SMALL
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an",
           "-vf", f"fps={GRID_HZ},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, e = None, []
    while True:
        buf = p.stdout.read(w * h)
        if len(buf) < w * h:
            break
        f = np.frombuffer(buf, np.uint8).reshape(h, w)
        e.append(0.0 if prev is None else float(np.mean(cv2.absdiff(f, prev))))
        prev = f
    p.wait()
    e = np.array(e)
    np.save(cache, e)
    return e


def grab_frame(path, t, width=320):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(path),
                          "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True)
    if not out.stdout:
        return None
    return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def cut_video(src, dst, t0, t1, fps, encoder):
    enc = (["-c:v", "h264_videotoolbox", "-b:v", "25M"] if encoder == "videotoolbox"
           else ["-c:v", "libx264", "-crf", "12", "-preset", "fast"])
    # high quality on purpose: compression is one of the levels being tested (L5)
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
         "-an", "-vf", f"fps={fps}", "-pix_fmt", "yuv420p", *enc, str(dst)])


# ----------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--video", action="append", default=[], help="name=path, e.g. phone=IMG_1234.MOV")
    ap.add_argument("--offset", action="append", default=[], help="name=seconds (pc_time - video_time)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--task", help="task name (default: from the bag's folder/file name)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--search", type=float, default=15.0,
                    help="coarse search window +- seconds around the file's creation time")
    a = ap.parse_args()

    bag = Path(a.bag)
    markers = bag.with_suffix(".markers.csv")
    if not markers.exists():
        sys.exit(f"No markers file next to the bag: {markers}")
    task = a.task or bag.name.split("_take")[0]
    videos = dict(v.split("=", 1) for v in a.video)
    forced = {k: float(v) for k, v in (o.split("=", 1) for o in a.offset)}
    out_root = Path(a.out) / task
    cache = Path(a.out) / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    encoder = "videotoolbox" if has_encoder("h264_videotoolbox") else "libx264"

    print(f"Reading {bag.name} (pass 1: timing + motion) ...")
    bt, be, depth_t, info, calib = scan_bag(bag)
    if len(bt) == 0:
        sys.exit("No color frames in the bag.")
    runs = read_markers(markers)
    print(f"  {len(bt)} color / {len(depth_t)} depth frames, {bt[-1] - bt[0]:.0f} s, {len(runs)} runs marked")
    grid_t, grid_e = to_grid(bt, be, bt[0], bt[-1])   # Xtion motion on the 30 Hz grid (pc time)

    # ---- per-video offsets ----------------------------------------------------
    # Each video file has ONE offset (its clock vs the lab PC's): pc_time = video_time + offset.
    # Method 1 (preferred): match the pattern of claps - audio spikes in the video vs. the
    #   clap spikes before each run in the Xtion bag. Unaffected by camera shake.
    # Method 2 (fallback): match the motion over the span of the runs.
    claps = bag_claps(grid_t, grid_e, runs)
    span0 = max(grid_t[0], min(t for _, t, _ in runs) - 6.0)
    span1 = min(grid_t[-1], max((e if e else grid_t[-1]) for _, _, e in runs) + 1.0)
    a0, a1 = np.searchsorted(grid_t, [span0, span1])
    span_e, span_t0 = grid_e[a0:a1], grid_t[a0]
    sync = {}
    for name, path in videos.items():
        print(f"Syncing {name} ({Path(path).name}) ...")
        ve = video_motion(path, cache)
        vinfo = ffprobe(path)
        start_time = float(vinfo["format"].get("start_time", 0) or 0)
        vdur = len(ve) / GRID_HZ
        created = creation_epoch(vinfo)
        # plausible offsets from the file's creation time (start or end of recording), wide
        prior = ([(g - 300, g + 300) for g in (created, created - vdur)] if created else None)
        if name in forced:
            sync[name] = {"path": path, "coarse_offset": forced[name], "motion": ve,
                          "method": "forced", "detail": "given with --offset"}
            print(f"  forced offset {forced[name]:.3f}")
            continue

        on_t, _ = audio_onsets(path, cache)
        m = match_claps(claps, on_t, prior)
        if m:
            off, k, spread, per = m
            ok = spread < 0.1
            print(f"  audio: {k}/{len(claps)} claps matched, spread {1000 * spread:.0f} ms -> offset {off:.3f}"
                  + ("" if ok else "   <-- spread is large, check the check sheets"))
            sync[name] = {"path": path, "coarse_offset": off, "motion": ve, "method": "audio",
                          "detail": f"{k}/{len(claps)} claps, spread {1000 * spread:.0f} ms",
                          "onsets": {n: claps[n] - v for n, v in per.items()}}   # video time of each clap
            continue
        print(f"  audio: couldn't match the claps ({len(on_t)} sound onsets found) - trying motion")

        # motion over the runs' span; partial overlap with the video is fine
        if created is not None:
            win = int(a.search * GRID_HZ)
            windows = [(int(round((span_t0 - g) * GRID_HZ)) - win, int(round((span_t0 - g) * GRID_HZ)) + win)
                       for g in (created, created - vdur)]
        else:
            windows = [(0, len(ve))]
        lags, sc = np.array([], int), np.array([])
        for lo, hi in windows:
            l_, s_ = lag_scores(span_e, ve, lo, hi)
            lags, sc = np.concatenate([lags, l_]), np.concatenate([sc, s_])
        if len(sc) == 0:
            print(f"  !! couldn't place the runs inside {name}'s video (is it the right file? does it "
                  f"cover all runs?). Skipping {name}; you can force it with --offset {name}=...")
            continue
        i = int(np.argmax(sc))
        lag, score = int(lags[i]), float(sc[i])
        far = np.abs(lags - lag) > 2 * GRID_HZ
        if far.any() and sc[far].max() > 0.9 * score:
            j = np.where(far)[0][int(np.argmax(sc[far]))]
            print(f"  !! AMBIGUOUS: offset {span_t0 - lags[j] / GRID_HZ:.3f} matches almost as well "
                  f"({sc[j]:.2f} vs {score:.2f}) - check the sheets")
        coarse = span_t0 - lag / GRID_HZ
        print(f"  motion: offset {coarse:.3f} (match score {score:.2f})")
        sync[name] = {"path": path, "coarse_offset": coarse, "motion": ve, "method": "motion",
                      "detail": f"motion score {score:.2f}"}

    # ---- refine the bag <-> video link ----------------------------------------------
    # Audio syncs the videos to EACH OTHER very precisely, but the bag has no audio: its
    # clap times come from motion and can be a few frames off - by the same amount for
    # every video. So: try small shifts (+-0.6 s) of all audio offsets together and keep the
    # one where the bag's motion best matches the most similar video (usually the iPad,
    # next to the Xtion).
    aud = [k for k, v in sync.items() if v["method"] == "audio"]
    if aud:
        best = None
        for k in aud:
            ve, off = sync[k]["motion"], sync[k]["coarse_offset"]
            lag0 = int(round((span_t0 - off) * GRID_HZ))
            lags, sc = lag_scores(span_e, ve, lag0 - 18, lag0 + 18)
            if len(sc):
                i = int(np.argmax(sc))
                delta = (lag0 - lags[i]) / GRID_HZ          # how much to add to the offset
                if best is None or sc[i] > best[1]:
                    best = (k, float(sc[i]), float(delta))
        if best and best[1] > 0.3:
            k, score, delta = best
            print(f"Refined the bag alignment using {k} (motion score {score:.2f}): "
                  f"all audio offsets shifted {1000 * delta:+.0f} ms")
            for kk in aud:
                sync[kk]["coarse_offset"] += delta
                sync[kk]["detail"] += f", bag link {1000 * delta:+.0f} ms via {k}"
        else:
            print("!! Couldn't refine the bag alignment by motion (no good match) - the offsets may be"
                  " a few frames off. Check the check sheets.")
    # The audio knows when each clap really happened: move the bag clap times there
    # (used for camera.yaml and to center the check sheets).
    for n in list(claps):
        ts_ = [sync[kk]["onsets"][n] + sync[kk]["coarse_offset"] for kk in aud if n in sync[kk]["onsets"]]
        if ts_:
            claps[n] = float(np.median(ts_))

    # ---- per run: plan the Xtion frames, write them in one pass ----------------
    out_root.mkdir(parents=True, exist_ok=True)
    plan, meta = {}, {}
    for n, t_start, t_end in runs:
        t_end = t_end if t_end is not None else float(bt[-1])
        rdir = out_root / f"run{n:02d}"
        if rdir.exists():
            shutil.rmtree(rdir)
        (rdir / "xtion" / "rgb").mkdir(parents=True)
        (rdir / "xtion" / "depth").mkdir(parents=True)
        sel = np.where((bt >= t_start) & (bt <= t_end))[0]
        rgb_index, depth_map, rows = {}, {}, []
        for k, i in enumerate(sel):
            s_ = float(bt[i])
            rgb_index[round(s_, 6)] = k
            dt = ""
            if len(depth_t):
                j = int(np.argmin(np.abs(depth_t - s_)))
                depth_map.setdefault(round(float(depth_t[j]), 6), []).append(k)
                dt = f"{depth_t[j]:.6f}"
            rows.append(f"{k},{s_:.6f},{dt}")
        (rdir / "xtion" / "frames.csv").write_text("index,rgb_time,depth_time\n" + "\n".join(rows) + "\n")
        plan[rdir] = (t_start, t_end, rgb_index, depth_map)
        clap_t = claps.get(n, t_start)       # the clap before this run (bag time)
        meta[rdir] = (n, t_start, t_end, clap_t, len(sel))
    print("Writing Xtion frames (pass 2) ...")
    write_frames(bag, plan)
    CHECK_DT = (-0.2, 0.0, 0.2)     # check sheet rows: just before, at, just after the clap
    clap_frames = bag_frames_at(bag, {(rdir, dt): m[3] + dt for rdir, m in meta.items() for dt in CHECK_DT}, bt)

    for rdir, (n, t_start, t_end, clap_t, k) in meta.items():
        cam = {"task": task, "run": n, "source_bag": bag.name, "frames": k,
               "start_pc_time": t_start, "end_pc_time": t_end, "clap_pc_time": clap_t}
        if info:
            cam.update(info)
        if calib:
            cam["base_link_to_camera_link"] = calib
        (rdir / "xtion" / "camera.yaml").write_text(yaml.safe_dump(cam, sort_keys=False))

        # fine sync each video on this run's window (clap included), then cut
        run_sync = {"task": task, "run": n, "start_pc_time": t_start, "end_pc_time": t_end,
                    "clap_pc_time": clap_t, "cameras": {}}
        blank = np.zeros((240, 320, 3), np.uint8)
        thumbs = [("xtion", [cv2.resize(clap_frames.get((rdir, dt), blank), (320, 240)) for dt in CHECK_DT])]
        for name, sv in sync.items():
            ve, coarse = sv["motion"], sv["coarse_offset"]
            if sv["method"] in ("audio", "forced"):
                off, score = coarse, None      # one clock offset for the whole file
            else:
                b0, b1 = np.searchsorted(grid_t, [t_start - 6.0, min(t_end + 1.0, grid_t[-1])])
                tmpl, seg_t0 = grid_e[b0:b1], grid_t[b0]
                lag0 = int(round((seg_t0 - coarse) * GRID_HZ))
                lag, score = best_lag(tmpl, ve, lag0 - 45, lag0 + 45)   # +-1.5 s
                off = seg_t0 - lag / GRID_HZ if lag is not None else coarse
            v0, v1 = t_start - off, t_end - off
            dst = rdir / f"{name}.mp4"
            cut_video(sv["path"], dst, v0, v1, a.fps, encoder)
            run_sync["cameras"][name] = {"source": str(sv["path"]), "offset": float(off),
                                         "video_start": float(v0), "video_end": float(v1),
                                         "method": sv["method"], "detail": sv["detail"],
                                         "match_score": None if score is None else round(float(score), 3)}
            col = []
            for dt in CHECK_DT:
                th = grab_frame(sv["path"], clap_t + dt - off)
                col.append(blank if th is None else cv2.resize(th, (320, int(320 * th.shape[0] / th.shape[1]))))
            thumbs.append((name, col))
            flag = "" if score is None or score > 0.3 else "   <-- low score, check check.jpg"
            how = sv["method"] if score is None else f"motion score {score:.2f}"
            print(f"  run {n:02d} {name:6s} offset {off:.3f}  ({how}){flag}")
        (rdir / "sync.yaml").write_text(yaml.safe_dump(run_sync, sort_keys=False))

        # check sheet: rows = 0.2 s before / at / after the clap, columns = cameras.
        # Every column should show the same instant in each row.
        cols = []
        for name, imgs in thumbs:
            parts = []
            for dt, t in zip(CHECK_DT, imgs):
                tile = np.zeros((240 + 24, 320, 3), np.uint8)
                tile[24:24 + min(240, t.shape[0])] = t[:240]
                cv2.putText(tile, f"{name}  clap {dt:+.1f}s", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                parts.append(tile)
            cols.append(np.vstack(parts))
        sheet = np.hstack(cols)
        cv2.putText(sheet, f"{task} run {n:02d}", (6, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(str(rdir / "check.jpg"), sheet)
        print(f"  run {n:02d}: {k} Xtion frames ({t_end - t_start:.1f} s) -> {rdir}")

    print(f"\nDone. Open each runs/{task}/runXX/check.jpg - all views should show the same clap moment.")
    if sync:
        print("Offsets (the same device should give nearly the same offset for every task in a session):")
        print("  " + "  ".join(f"--offset {k}={v['coarse_offset']:.3f}" for k, v in sync.items()))


if __name__ == "__main__":
    main()