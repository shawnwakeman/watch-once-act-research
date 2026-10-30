#!/usr/bin/env python3
"""Touch-point camera calibration: finds base_link -> camera_link from matching points.

For each point (6-8, spread over the work area, ideally a few at different heights):
  1. Touch it with the gripper tip (gripper pointing straight down) -> arm gives its
     position in base_link.
  2. Move the arm out of the camera's view, then click the SAME point in RViz with
     "Publish Point" -> camera gives its position in camera_link (from depth).
Then Kabsch finds the rigid transform between the two point sets, prints the error at
every point, and saves it in the same format as easy_handeye, so publish_calibration.py
works unchanged.

Needs running: start_driver.sh, start_camera.sh, and RViz with
  Fixed Frame = camera_link, PointCloud2 = /camera/depth_registered/points.
Do NOT run publish_calibration.py or calibrate.sh at the same time.

Before the points, a tip check: touch the BARE TABLE once. If tool_frame isn't exactly at
the fingertips, every touch is off by the same amount (invisible in the fit error), so the
script measures that offset and corrects for it. You'll be asked whether the arm base sits
directly on the table, or how thick its plate/mount is.

Keep the gripper pointing straight down with the SAME wrist twist for every touch - the
script warns if tilt > 5 deg or twist changes > 10 deg.

Usage (inside the container):
    python3 touch_calibrate.py
    TIP_FRAME=end_effector_link python3 touch_calibrate.py   # if tool_frame isn't the tip
    TIP_DZ=0.018 python3 touch_calibrate.py                  # skip the tip check, use this offset
"""
import os
import sys
import datetime

import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
import yaml
from geometry_msgs.msg import PointStamped

BASE = "base_link"
CAMERA = "camera_link"
TIP = os.environ.get("TIP_FRAME", "tool_frame")
HERE = os.path.dirname(os.path.abspath(__file__))


def kabsch(P, Q):
    """Best rigid R, t with Q ~= R @ P + t  (P = camera points, Q = base points)."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tip_pose(buf):
    """Tip position, tilt from straight-down (deg) and wrist twist (deg) in base_link."""
    tr = buf.lookup_transform(BASE, TIP, rospy.Time(0), rospy.Duration(1.0)).transform
    R = quat_to_rot(tr.rotation)
    approach = R[:, 2]                                   # tool z-axis = pointing direction
    tilt = float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1))))
    twist = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))  # heading of the tool x-axis
    return np.array([tr.translation.x, tr.translation.y, tr.translation.z]), tilt, twist


def tip_check(buf):
    """Returns TIP_DZ (m) to add to every arm z so it matches the real fingertip contact."""
    if "TIP_DZ" in os.environ:
        dz = float(os.environ["TIP_DZ"])
        print(f"Using TIP_DZ = {dz * 1000:+.1f} mm from the environment.\n")
        return float(dz), "env"
    print("TIP CHECK: lower the closed gripper, pointing straight down, until the fingertips")
    if input("just touch the BARE TABLE (no sticker). Press Enter (or 's' to skip): ").strip().lower() == "s":
        print("  Skipped - no tip correction.\n")
        return 0.0, "skipped"
    pos, tilt, _ = tip_pose(buf)
    print(f"  {TIP} z at table contact = {pos[2] * 1000:+.1f} mm  (tilt {tilt:.1f} deg)")
    if tilt > 5:
        print("  !! Gripper isn't straight down - the check is less reliable. Consider redoing it.")
    ans = input("  Does the arm base sit DIRECTLY on this table? [y = yes / n = it's on a plate]: ").strip().lower()
    if ans.startswith("n"):
        plate = float(input("  Plate/mount thickness under the base, in mm: ").strip()) / 1000
        table_z = -plate
    else:
        table_z = 0.0
    dz = table_z - pos[2]
    print(f"  -> Table is at z = {table_z * 1000:+.1f} mm in {BASE}; tip correction = {dz * 1000:+.1f} mm")
    if abs(dz) < 0.003:
        print("     (under 3 mm - tool_frame is effectively at the fingertips)")
    print()
    return float(dz), f"table touch, base on {'plate ' + str(-table_z * 1000) + ' mm' if table_z else 'table'}"


def check_last_point(base_pts, cam_pts):
    """Distances between points don't depend on the calibration, so the arm's and the
    camera's measurements must agree. Flags a bad click/touch right away."""
    if len(base_pts) < 2:
        print("  (consistency check starts at point 2)\n")
        return
    b, c = np.array(base_pts), np.array(cam_pts)
    db = np.linalg.norm(b[:-1] - b[-1], axis=1)
    dc = np.linalg.norm(c[:-1] - c[-1], axis=1)
    diff = 100 * np.abs(db - dc)
    # Allow 2 cm, or 5% of the distance for far-apart points: the Xtion with default
    # intrinsics measures long distances a few % short, which isn't a bad click.
    limit = np.maximum(2.0, 5.0 * db)
    worst = int(np.argmax(diff - limit))
    print("  Distance check vs earlier points (cm off): " + ", ".join(f"{d:.1f}" for d in diff))
    if diff[worst] > limit[worst]:
        print(f"  !! Mismatch of {diff[worst]:.1f} cm with point {worst + 1}. Probably this click or touch"
              f" (or point {worst + 1}) is off - type 'u' to undo this point and redo it.\n")
    else:
        print("  OK - consistent with earlier points.\n")


def fit_report_quiet(P, Q):
    R, t = kabsch(P, Q)
    return float(np.sqrt(((100 * np.linalg.norm((R @ P.T).T + t - Q, axis=1)) ** 2).mean()))


def fit_report(P, Q, labels):
    R, t = kabsch(P, Q)
    err = 100 * np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    rms = float(np.sqrt((err ** 2).mean()))
    print("Per-point error (cm): " + ", ".join(f"#{l}:{e:.1f}" for l, e in zip(labels, err)))
    print(f"RMS error: {rms:.2f} cm   (under ~1 cm great, 1-2 cm OK, over 2 cm redo/check)")
    return R, t, err, rms


def main():
    rospy.init_node("touch_calibrate")
    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference or it stops listening
    rospy.sleep(1.0)

    try:
        buf.lookup_transform(BASE, TIP, rospy.Time(0), rospy.Duration(2.0))
    except Exception as e:
        sys.exit(f"Can't find {BASE} -> {TIP}: {e}\nTry TIP_FRAME=end_effector_link")

    tip_dz, tip_src = tip_check(buf)
    twist0 = None

    base_pts, cam_pts = [], []
    print(f"\nTip frame: {TIP}.  Aim for 6-8 points spread over the work area.")
    print("Type 'd' + Enter when done (min 4), 'u' to undo the last point.\nNumber your stickers and touch/click them in the same order.\n")

    while not rospy.is_shutdown():
        n = len(base_pts) + 1
        cmd = input(f"[Point {n}] Touch it with the gripper tip (pointing down), then Enter: ").strip().lower()
        if cmd == "d":
            if len(base_pts) >= 4:
                break
            print("  Need at least 4 points.")
            continue
        if cmd == "u":
            if base_pts:
                base_pts.pop(); cam_pts.pop()
                print("  Removed last point.")
            continue

        bp, tilt, twist = tip_pose(buf)
        bp[2] += tip_dz
        print(f"  Arm: tip at ({bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}) m in {BASE}"
              f"  (tilt {tilt:.1f} deg, twist {twist:.0f} deg)")
        if twist0 is None:
            twist0 = twist
        dtw = abs((twist - twist0 + 180) % 360 - 180)
        if tilt > 5 or dtw > 10:
            why = []
            if tilt > 5:
                why.append(f"tilted {tilt:.0f} deg from straight down")
            if dtw > 10:
                why.append(f"wrist twist changed {dtw:.0f} deg from point 1")
            if input(f"  !! Gripper {' and '.join(why)}. Redo this touch? [Y/n] ").strip().lower() != "n":
                continue

        print("  Now move the arm OUT of the camera's view, and click the same point in RViz (Publish Point)...")
        clicked = rospy.wait_for_message("/clicked_point", PointStamped)
        try:
            cp = buf.transform(clicked, CAMERA, rospy.Duration(1.0)).point
        except Exception as e:
            print(f"  Couldn't convert click to {CAMERA} ({e}). Is RViz Fixed Frame = camera_link? Point skipped.")
            continue
        cam = np.array([cp.x, cp.y, cp.z])
        print(f"  Camera: ({cam[0]:.3f}, {cam[1]:.3f}, {cam[2]:.3f}) m in {CAMERA}  "
              f"[{100 * np.linalg.norm(cam):.0f} cm from camera]")
        base_pts.append(bp)
        cam_pts.append(cam)
        check_last_point(base_pts, cam_pts)

    P, Q = np.array(cam_pts), np.array(base_pts)
    labels = list(range(1, len(P) + 1))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    d = os.path.join(HERE, "calibration")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"touch_points_{stamp}.yaml"), "w") as f:   # raw data, always kept
        yaml.safe_dump({"tip_frame": TIP, "tip_dz_m": float(tip_dz), "tip_dz_source": tip_src,
                        "note": "points_base already include tip_dz",
                        "points_base": Q.tolist(), "points_camera": P.tolist()}, f)

    print()
    R, t, err, rms = fit_report(P, Q, labels)
    # Offer to drop the worst point while the fit is poor and enough points remain
    while rms > 1.5 and len(P) > 4:
        # leave-one-out: which single point, if removed, helps most?
        best = min(range(len(P)), key=lambda i: fit_report_quiet(np.delete(P, i, 0), np.delete(Q, i, 0)))
        if input(f"\nDrop point #{labels[best]} and refit? [y/N] ").strip().lower() != "y":
            break
        P, Q = np.delete(P, best, 0), np.delete(Q, best, 0)
        labels.pop(best)
        R, t, err, rms = fit_report(P, Q, labels)

    print(f"Camera in {BASE}: x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f} m "
          f"({t[0] / 0.0254:.1f}, {t[1] / 0.0254:.1f}, {t[2] / 0.0254:.1f} in)")
    err_cm = err

    if input("\nSave this calibration? [y/N] ").strip().lower() != "y":
        return
    qx, qy, qz, qw = rot_to_quat(R)
    out = {
        "eye_on_hand": False,
        "robot_base_frame": BASE,
        "tracking_base_frame": CAMERA,
        "method": f"touch-point Kabsch, {len(P)} points, tip={TIP}, tip_dz={tip_dz * 1000:+.1f} mm ({tip_src})",
        "rms_error_cm": float(np.sqrt((err_cm ** 2).mean())),
        "points_base": Q.tolist(),
        "points_camera": P.tolist(),
        "transformation": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2]),
                           "qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw)},
    }
    for name in (f"touch_{stamp}.yaml", "current.yaml"):
        with open(os.path.join(d, name), "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
    print(f"Saved calibration/touch_{stamp}.yaml and calibration/current.yaml")
    print("Next: python3 publish_calibration.py, then check RViz with Fixed Frame = base_link.")


if __name__ == "__main__":
    main()