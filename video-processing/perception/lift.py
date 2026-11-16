#!/usr/bin/env python3
"""Stage 2 on one run: tracked pixels + depth -> 3D points -> object pose per frame.

  back-project:  X = (u - cx) Z / fx,  Y = (v - cy) Z / fy   (camera optical frame)
  Kabsch:        frame 0's points -> frame t's points  =>  R_t, t_t  (robust: outliers dropped)
  pose:          object frame at the frame-0 centroid, axes = base_link axes
  release:       a fast, lasting fall in height = the object was dropped -> trajectory ends there
  smooth, then express in base_link (calibration) and relative to the start / the target

Usage (after perceive.py, L0 = real Xtion depth):
  python3 lift.py runs/pick_place/run01
  python3 lift.py runs/bowl_drop/run02          # uses target_mask.png if perceive found one

Writes runs/.../perception/xtion/:
  trajectory.npz   T_base (T,4,4) object pose in base_link; T_rel (T,4,4) relative to its
                   start pose; times (T,); n_used (T,); rms_mm (T,) Kabsch fit error
  trajectory.yaml  start / end position, displacement, rotation, fit quality, target centroid
  trajectory.png   top-down path (base_link x-y), height over time, fit error over time
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# openni2 / rgbd_launch defaults, used only if the bag didn't carry the camera's own frames
DEFAULT_LINK_TO_RGB = [0.0, -0.045, 0.0, 0.0, 0.0, 0.0, 1.0]
DEFAULT_RGB_TO_OPTICAL = [0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5]


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_from(xyzq):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(*xyzq[3:])
    T[:3, 3] = xyzq[:3]
    return T


def base_to_optical(cam):
    """base_link -> camera_rgb_optical_frame, from the transforms saved with the run."""
    c = cam.get("base_link_to_camera_link")
    if c is None:
        raise SystemExit("No base_link->camera_link calibration in camera.yaml (was publish_calibration.py running?)")
    T = T_from([c["x"], c["y"], c["z"], c["qx"], c["qy"], c["qz"], c["qw"]])
    st = cam.get("tf_static", {})
    link_rgb = st.get("camera_link->camera_rgb_frame", DEFAULT_LINK_TO_RGB)
    rgb_opt = st.get("camera_rgb_frame->camera_rgb_optical_frame", DEFAULT_RGB_TO_OPTICAL)
    if "camera_link->camera_rgb_frame" not in st:
        print("  (camera frames not in this run's camera.yaml - using the openni2 defaults; "
              "re-run prep_runs.py to store the exact ones)")
    return T @ T_from(link_rgb) @ T_from(rgb_opt)


def kabsch(P, Q):
    """R, t minimizing sum ||R p + t - q||^2."""
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def robust_kabsch(P, Q, iters=3, floor=0.004):
    """Kabsch, then drop points whose residual is > 2.5x the median (and > 4 mm), refit."""
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        R, t = kabsch(P[keep], Q[keep])
        r = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        new = r <= max(floor, 2.5 * np.median(r[keep]))
        if (new == keep).all():
            break
        keep = new
    R, t = kabsch(P[keep], Q[keep])
    r = np.linalg.norm((R @ P[keep].T).T + t - Q[keep], axis=1)
    return R, t, keep, float(np.sqrt(np.mean(r ** 2)))


def sample_depth(depth_mm, u, v, r=2):
    """Median of valid depth in a (2r+1)^2 patch, in meters (nan if none)."""
    h, w = depth_mm.shape
    out = np.full(len(u), np.nan)
    for i, (x, y) in enumerate(zip(np.round(u).astype(int), np.round(v).astype(int))):
        if 0 <= x < w and 0 <= y < h:
            patch = depth_mm[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].ravel()
            patch = patch[patch > 0]
            if len(patch):
                out[i] = np.median(patch) / 1000.0
    return out


def smooth_poses(Rs, ts, half=2):
    """Centered moving average: positions averaged, rotations averaged then re-orthonormalized."""
    n = len(ts)
    Rs2, ts2 = np.empty_like(Rs), np.empty_like(ts)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        ts2[i] = ts[a:b].mean(0)
        U, _, Vt = np.linalg.svd(Rs[a:b].sum(0))
        Rs2[i] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return Rs2, ts2


def find_release(z, dt, fall_speed=0.5, min_drop=0.03, settle_s=0.5, start_speed=0.2):
    """The object was let go: a fast fall (> fall_speed m/s down) that lasts (the height
    0.5 s later is > min_drop below where the fall started). A hand lowering the cube moves
    ~0.05 m/s; a 10 cm free fall reaches ~1.4 m/s. Returns the last frame before the fall
    started, or None."""
    vz = np.convolve(np.gradient(z) / dt, np.ones(3) / 3, mode="same")
    settle = max(1, int(round(settle_s / dt)))
    for f in np.nonzero(vz < -fall_speed)[0]:
        s = f
        while s > 0 and vz[s - 1] < -start_speed:      # walk back to where the fall began
            s -= 1
        after = z[f + 1:f + 1 + settle]
        if len(after) and np.median(after) < z[s] - min_drop:
            return max(0, s - 2)              # 2-frame margin: still in the hand
    return None


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--max-fit-mm", type=float, default=6.0,
                    help="frames with a worse Kabsch fit are treated as unreliable and interpolated")
    ap.add_argument("--min-visible", type=float, default=0.5,
                    help="end the trajectory once fewer than this fraction of points stay visible")
    ap.add_argument("--no-release-cut", action="store_true",
                    help="keep frames after a detected drop (by default the trajectory ends at the release)")
    a = ap.parse_args()

    run = Path(a.run_dir)
    pdir = run / "perception" / "xtion"
    cam = yaml.safe_load((run / "xtion" / "camera.yaml").read_text())
    K = np.array(cam["K"]).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    trk = np.load(pdir / "tracks.npz")
    tracks, visible = trk["tracks"], trk["visible"]
    T_frames, N = tracks.shape[:2]
    depth_files = sorted((run / "xtion" / "depth").glob("*.png"))
    frames_csv = np.genfromtxt(run / "xtion" / "frames.csv", delimiter=",", skip_header=1)
    times = frames_csv[:T_frames, 1] - frames_csv[0, 1]
    T_bo = base_to_optical(cam)

    # 3D points per frame (camera optical frame), nan where hidden / no depth
    P = np.full((T_frames, N, 3), np.nan)
    for t in range(T_frames):
        d = cv2.imread(str(depth_files[t]), cv2.IMREAD_UNCHANGED)
        u, v = tracks[t, :, 0], tracks[t, :, 1]
        Z = sample_depth(d, u, v)
        Z[~visible[t]] = np.nan
        P[t] = np.stack([(u - cx) * Z / fx, (v - cy) * Z / fy, Z], 1)
    ok0 = ~np.isnan(P[0, :, 2])
    print(f"{run}: {T_frames} frames, {N} tracked points, {ok0.sum()} with depth in frame 0")
    if ok0.sum() < a.min_points:
        raise SystemExit("Too few points with depth in frame 0 - check the mask / depth.")

    # Kabsch frame 0 -> t (in the camera frame)
    Rs, tvec = np.tile(np.eye(3), (T_frames, 1, 1)), np.zeros((T_frames, 3))
    n_used, rms = np.zeros(T_frames, int), np.full(T_frames, np.nan)
    for t in range(T_frames):
        ok = ok0 & ~np.isnan(P[t, :, 2])
        if ok.sum() < a.min_points:          # too few: hold the previous pose (logged)
            Rs[t], tvec[t] = Rs[t - 1], tvec[t - 1]
            continue
        R, tt, keep, e = robust_kabsch(P[0, ok], P[t, ok])
        Rs[t], tvec[t], n_used[t], rms[t] = R, tt, int(keep.sum()), e * 1000
    held = int((n_used == 0).sum())

    # Release: if the object was dropped (bowl_drop), everything after the hand lets go is
    # the fall + tumble inside the box - not part of the demonstrated motion, and the fit
    # is bad there (the cube spins, faces get hidden by the box rim). End at the release.
    R_bo, t_bo = T_bo[:3, :3], T_bo[:3, 3]
    c0 = np.nanmean(P[0, ok0], 0)
    pos_raw = ((Rs @ c0) + tvec) @ R_bo.T + t_bo          # unsmoothed object position, base_link
    times_full, rms_full, z_full = times.copy(), rms.copy(), pos_raw[:, 2].copy()
    dt = float(np.median(np.diff(times)))
    release = find_release(pos_raw[:, 2], dt)
    landed = None
    if release is not None:
        tail = slice(max(release + 1, T_frames - int(round(0.5 / dt))), T_frames)
        ok_tail = n_used[tail] >= a.min_points
        if ok_tail.any():
            landed = np.median(pos_raw[tail][ok_tail], 0)    # where it came to rest (rough: fit is poor there)
        print(f"  released at frame {release} ({times[release]:.2f} s), "
              f"height {pos_raw[release, 2] * 100:.1f} cm -> fell to {z_full[release + 1:].min() * 100:.1f} cm")
        if not a.no_release_cut:
            T_frames = release + 1
            Rs, tvec, n_used, rms, times, visible = (Rs[:T_frames], tvec[:T_frames], n_used[:T_frames],
                                                     rms[:T_frames], times[:T_frames], visible[:T_frames])

    # Frame quality: a frame is unreliable if too few points were fitted or the fit error
    # is high (points slid onto the hand/table, depth taken mid-motion, rim occlusion).
    med = float(np.nanmedian(rms))
    bad_thr = max(a.max_fit_mm, 2.5 * med)
    vis_frac = visible.mean(1)
    good = (n_used >= a.min_points) & (rms <= bad_thr) & (vis_frac >= a.min_visible)
    good[0] = True

    # If the object ends up hidden or badly fitted for good (dropped into a box), the last
    # poses are guesses: end the trajectory at the last reliable frame (for bowl_drop,
    # about the release).
    last_good = int(np.nonzero(good)[0].max())
    truncated = last_good < T_frames - 1 and (~good[last_good + 1:]).mean() > 0.5
    if truncated:
        print(f"  object hidden / badly fitted after frame {last_good} of {T_frames} "
              f"({vis_frac[last_good + 1:].mean():.0%} of points visible after) - trajectory ends there")
        T_frames = last_good + 1
        Rs, tvec, n_used, rms, times, good = (Rs[:T_frames], tvec[:T_frames], n_used[:T_frames],
                                              rms[:T_frames], times[:T_frames], good[:T_frames])

    # Unreliable frames in between: replace their pose by interpolating between the
    # nearest reliable frames on each side (instead of trusting a bad fit).
    rejected = np.nonzero(~good)[0]
    gi = np.nonzero(good)[0]
    for t in rejected:
        i0 = gi[gi < t].max()
        later = gi[gi > t]
        if len(later) == 0:
            Rs[t], tvec[t] = Rs[i0], tvec[i0]
            continue
        i1 = later.min()
        w = (t - i0) / (i1 - i0)
        rv, _ = cv2.Rodrigues(Rs[i0].T @ Rs[i1])                 # relative rotation as a vector
        Rs[t] = Rs[i0] @ cv2.Rodrigues(rv * w)[0]
        tvec[t] = (1 - w) * tvec[i0] + w * tvec[i1]
    if len(rejected):
        print(f"  {len(rejected)} of {T_frames} frames unreliable (fit > {bad_thr:.1f} mm or too few points) "
              "- interpolated")

    # object pose in the camera frame: origin = frame-0 centroid, moved by (R_t, t_t)
    Rs, pos_cam = smooth_poses(Rs, (Rs @ c0) + tvec)

    # into base_link; object axes start aligned with base_link
    T_base = np.tile(np.eye(4), (T_frames, 1, 1))
    for t in range(T_frames):
        T_base[t, :3, :3] = R_bo @ Rs[t] @ R_bo.T        # rotation of the object, in base axes
        T_base[t, :3, 3] = R_bo @ pos_cam[t] + t_bo
    T_rel = np.linalg.inv(T_base[0]) @ T_base           # relative to the start pose

    summary = {
        "run": str(run), "frames": int(T_frames), "duration_s": float(times[-1]),
        "start_base_m": [round(float(v), 4) for v in T_base[0, :3, 3]],
        "end_base_m": [round(float(v), 4) for v in T_base[-1, :3, 3]],
        "displacement_m": [round(float(v), 4) for v in T_base[-1, :3, 3] - T_base[0, :3, 3]],
        "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(T_base[:, :3, 3], axis=0), axis=1))), 4),
        "end_rotation_deg": round(rot_deg(T_rel[-1, :3, :3]), 1),
        "fit_rms_mm_median": round(float(np.nanmedian(rms)), 2),
        "fit_rms_mm_p95": round(float(np.nanpercentile(rms, 95)), 2),
        "frames_rejected_interpolated": int(len(rejected)),
        "points_used_median": int(np.median(n_used[n_used > 0])) if (n_used > 0).any() else 0,
        "frames_held_too_few_points": held,
        "ended_early_at_frame": last_good if truncated else None,
        "release_frame": None if release is None else int(release),
        "release_time_s": None if release is None else round(float(times_full[release]), 3),
        "trajectory_ends_at_release": bool(release is not None and not a.no_release_cut),
    }
    if landed is not None:
        summary["landed_base_m"] = [round(float(v), 4) for v in landed]
    tmask = pdir / "target_mask.png"
    if tmask.exists():                                    # target centroid (frame 0) for two-object tasks
        m = cv2.imread(str(tmask), cv2.IMREAD_GRAYSCALE) > 0
        d0 = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
        ys, xs = np.nonzero(m & (d0 > 0))
        if len(xs):
            Z = d0[ys, xs] / 1000.0
            pc = np.stack([(xs - cx) * Z / fx, (ys - cy) * Z / fy, Z], 1)
            tc = R_bo @ np.median(pc, 0) + t_bo
            summary["target_centroid_base_m"] = [round(float(v), 4) for v in tc]
            summary["end_minus_target_m"] = [round(float(v), 4) for v in T_base[-1, :3, 3] - tc]
            if landed is not None:
                summary["landed_minus_target_m"] = [round(float(v), 4) for v in landed - tc]

    np.savez_compressed(pdir / "trajectory.npz", T_base=T_base, T_rel=T_rel, times=times,
                        n_used=n_used, rms_mm=rms, reliable=good)
    (pdir / "trajectory.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    xyz = T_base[:, :3, 3]
    ax[0].plot(xyz[:, 0], xyz[:, 1], "-")
    ax[0].plot(*xyz[0, :2], "go", label="start")
    ax[0].plot(*xyz[-1, :2], "rs", label="end")
    if "target_centroid_base_m" in summary:
        ax[0].plot(*summary["target_centroid_base_m"][:2], "k*", ms=12, label="target")
    ax[0].set_xlabel("base x (m)"); ax[0].set_ylabel("base y (m)"); ax[0].set_title("Top-down path")
    ax[0].axis("equal"); ax[0].legend()
    if release is not None:
        ax[1].plot(times_full, z_full * 100, color="0.75", label="after release (cut)")
        ax[2].plot(times_full, rms_full, color="0.75")
        for k in (1, 2):
            ax[k].axvline(times_full[release], color="m", lw=1, ls=":")
        if landed is not None:
            ax[0].plot(*landed[:2], "mx", ms=9, label="landed (rough)")
            ax[0].legend()
    ax[1].plot(times, xyz[:, 2] * 100); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("height z (cm)")
    ax[1].set_title("Height" + ("  (release = dotted line)" if release is not None else ""))
    ax[2].plot(times, rms)
    if len(rejected):
        ax[2].plot(times[rejected], rms[rejected], "r.", label="unreliable -> interpolated")
        ax[2].legend()
    ax[2].axhline(bad_thr, color="r", lw=0.8, ls="--")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("fit error (mm)")
    ax[2].set_title(f"Kabsch fit error (points used, median {summary['points_used_median']})")
    fig.tight_layout(); fig.savefig(pdir / "trajectory.png", dpi=120)

    print(yaml.safe_dump(summary, sort_keys=False))
    print(f"-> {pdir}/trajectory.npz, .yaml, .png")


if __name__ == "__main__":
    main()