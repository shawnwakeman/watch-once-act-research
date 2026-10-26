#!/usr/bin/env python3
"""Publish the saved hand-eye calibration as a fixed tf link: base_link -> camera_link.

After this runs, every camera point can be converted into arm coordinates with tf.
Leave it running (it's tiny).

Usage (inside the container):
    python3 publish_calibration.py                      # loads calibration/current.yaml
    python3 publish_calibration.py calibration/handeye_2026-10-26_1400.yaml
"""
import os
import sys

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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "calibration", "current.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)

    tf = find_key(data, "transformation")
    parent = find_key(data, "robot_base_frame") or "base_link"
    child = find_key(data, "tracking_base_frame") or "camera_link"
    if tf is None:
        sys.exit(f"No 'transformation' found in {path}")

    rospy.init_node("publish_calibration")
    msg = TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = tf["x"]
    msg.transform.translation.y = tf["y"]
    msg.transform.translation.z = tf["z"]
    msg.transform.rotation.x = tf["qx"]
    msg.transform.rotation.y = tf["qy"]
    msg.transform.rotation.z = tf["qz"]
    msg.transform.rotation.w = tf["qw"]

    broadcaster = tf2_ros.StaticTransformBroadcaster()  # keep a reference while spinning
    broadcaster.sendTransform(msg)
    rospy.loginfo("Publishing %s -> %s from %s", parent, child, path)
    rospy.loginfo("  xyz = (%.3f, %.3f, %.3f) m", tf["x"], tf["y"], tf["z"])
    rospy.spin()


if __name__ == "__main__":
    main()
