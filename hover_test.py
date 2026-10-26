#!/usr/bin/env python3
"""Calibration check: click a point in RViz, and the gripper hovers above it.

Needs running: start_driver.sh, start_camera.sh, publish_calibration.py.
In RViz: Fixed Frame = base_link, show the point cloud, use the "Publish Point"
tool (toolbar) and click a spot on the table. The script converts that camera
point into arm coordinates and moves the gripper to HOVER_M above it.

Measure the gap between the gripper tip and the clicked spot: under ~1-2 cm
sideways means the calibration is good.

Usage (inside the container):  python3 hover_test.py
    HOVER_M=0.15 python3 hover_test.py     # hover higher
"""
import math
import os
import sys

os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from geometry_msgs.msg import PointStamped

HOVER_M = float(os.environ.get("HOVER_M", "0.10"))
SPEED = 0.1
BASE = "base_link"
# Safety box in base_link (meters). Refuse targets outside it.
MIN_REACH, MAX_REACH = 0.15, 0.65
MIN_Z = 0.03


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("hover_test")

    arm = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
    arm.set_max_velocity_scaling_factor(SPEED)
    arm.set_max_acceleration_scaling_factor(SPEED)

    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference or it stops listening

    print(f"End effector: {arm.get_end_effector_link()}  |  hover height: {HOVER_M} m")
    print("Click a point in RViz with 'Publish Point'. Ctrl+C to quit.")

    while not rospy.is_shutdown():
        clicked = rospy.wait_for_message("/clicked_point", PointStamped)
        try:
            p = buf.transform(clicked, BASE, rospy.Duration(1.0)).point
        except Exception as e:
            print(f"Couldn't convert to {BASE}: {e}\n(Is publish_calibration.py running?)")
            continue

        reach = math.hypot(p.x, p.y)
        target_z = p.z + HOVER_M
        print(f"\nClicked point in {BASE}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}  (reach {reach:.2f} m)")
        if not (MIN_REACH <= reach <= MAX_REACH) or target_z < MIN_Z:
            print("Outside the safety box - pick a point closer to the arm.")
            continue

        if input(f"Move gripper to z={target_z:.3f} above it? [y/N] (gamepad RB = E-stop) ").lower() != "y":
            continue

        pose = arm.get_current_pose().pose          # keep current orientation
        pose.position.x, pose.position.y, pose.position.z = p.x, p.y, target_z
        arm.set_pose_target(pose)
        ok = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if not ok:
            print("Planning failed for that point.")
            continue
        now = arm.get_current_pose().pose.position
        print(f"Gripper now at x={now.x:.3f} y={now.y:.3f} z={now.z:.3f}. "
              "Measure the sideways gap to the clicked spot.")


if __name__ == "__main__":
    main()
