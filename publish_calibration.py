#!/usr/bin/env python3
"""Publish the saved camera calibration as a fixed tf link: base_link -> camera_link.

After this runs, every camera point can be converted into arm coordinates with tf.
Leave it running (it's tiny).

Usage (inside the container):
    python3 publish_calibration.py                      # loads calibration/current.yaml
    python3 publish_calibration.py calibration/touch_2026-10-28_1412.yaml
    python3 publish_calibration.py --tune               # nudge the position live, then save

Tune mode (watch RViz with Fixed Frame = base_link while you type):
    x 5      move the camera +5 mm along base x (red axis)     x -5  the other way
    y 5 / z 5      same for y (green) and z (blue, up)
    show     print the current offset
    s        save to calibration/current.yaml (keeps a backup + records the offset)
    q        quit without saving (the original stays on disk)
If the point cloud sits too LOW, raise the camera (z +). Moving the camera moves the
whole cloud the same way.
"""
import os
import shutil
import sys
import datetime

import numpy as np
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped

HERE = os.path.dirname(os.path.abspath(__file__))


def find_key(d, key):
    """Return d[key], looking one level down too (easy_handeye's format varies by version)."""
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return None


def make_msg(parent, child, tf, offset_m):
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = tf["x"] + offset_m[0]
    msg.transform.translation.y = tf["y"] + offset_m[1]
    msg.transform.translation.z = tf["z"] + offset_m[2]
    msg.transform.rotation.x = tf["qx"]
    msg.transform.rotation.y = tf["qy"]
    msg.transform.rotation.z = tf["qz"]
    msg.transform.rotation.w = tf["qw"]
    return msg


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("__")]   # ignore ROS remap args
    tune = "--tune" in args
    args = [a for a in args if a != "--tune"]
    path = args[0] if args else os.path.join(HERE, "calibration", "current.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)

    tf = find_key(data, "transformation")
    parent = find_key(data, "robot_base_frame") or "base_link"
    child = find_key(data, "tracking_base_frame") or "camera_link"
    if tf is None:
        sys.exit(f"No 'transformation' found in {path}")

    rospy.init_node("publish_calibration", disable_signals=tune)
    broadcaster = tf2_ros.StaticTransformBroadcaster()  # keep a reference while running
    offset = np.zeros(3)
    broadcaster.sendTransform(make_msg(parent, child, tf, offset))
    rospy.loginfo("Publishing %s -> %s from %s", parent, child, path)
    rospy.loginfo("  xyz = (%.3f, %.3f, %.3f) m", tf["x"], tf["y"], tf["z"])

    if not tune:
        rospy.spin()
        return

    print("\nTune mode: 'x 5' / 'y -3' / 'z 10' (mm), 'show', 's' save, 'q' quit.")
    while not rospy.is_shutdown():
        try:
            cmd = input("tune> ").strip().lower().split()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd[0] == "q":
            print("Quit without saving.")
            break
        if cmd[0] == "show":
            print(f"  offset (mm): x={1000 * offset[0]:+.0f} y={1000 * offset[1]:+.0f} z={1000 * offset[2]:+.0f}")
            continue
        if cmd[0] == "s":
            for k, v in zip("xyz", offset):
                tf[k] = float(tf[k] + v)
            data["manual_offset_mm"] = [round(float(1000 * v), 1) for v in
                                        np.array(data.get("manual_offset_mm", [0, 0, 0])) / 1000 + offset]
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            shutil.copy(path, path.replace(".yaml", f"_before_tune_{stamp}.yaml"))
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            offset[:] = 0
            print(f"  Saved to {path} (backup kept). Total manual offset now {data['manual_offset_mm']} mm.")
            continue
        if cmd[0] in ("x", "y", "z") and len(cmd) == 2:
            try:
                mm = float(cmd[1])
            except ValueError:
                print("  Use e.g. 'x 5' (millimeters)."); continue
            offset["xyz".index(cmd[0])] += mm / 1000.0
            broadcaster.sendTransform(make_msg(parent, child, tf, offset))
            print(f"  offset (mm): x={1000 * offset[0]:+.0f} y={1000 * offset[1]:+.0f} z={1000 * offset[2]:+.0f}")
            continue
        print("  Commands: 'x 5', 'y -3', 'z 10' (mm), 'show', 's', 'q'")
    rospy.signal_shutdown("done")


if __name__ == "__main__":
    main()