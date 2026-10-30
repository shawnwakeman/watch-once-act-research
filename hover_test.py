#!/usr/bin/env python3
"""Calibration check: click a point in RViz, and the gripper hovers above it.

Needs running: start_driver.sh, start_camera.sh, publish_calibration.py.
In RViz: Fixed Frame = base_link, show the point cloud, use the "Publish Point"
tool (toolbar) and click a spot on the table. The script converts that camera
point into arm coordinates and moves the gripper to HOVER_M above it.

Measure the gap between the gripper tip and the clicked spot: under ~1 cm
sideways means the calibration is good.

Repeat mode: after a move, press Enter to go to the SAME point again. The point is
stored in the camera's frame and re-converted with the calibration as it is NOW, so
you can nudge it with `publish_calibration.py --tune` in another terminal and see
the effect without clicking again.

Usage (inside the container):
    python3 hover_test.py
    HOVER_M=0.05 python3 hover_test.py      # hover lower (default 0.10 m)
    SPEED=0.3 python3 hover_test.py         # faster (default 0.1 = 10%)
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

HOVER_M = float(os.environ.get("HOVER_M", "0.70"))
SPEED = float(os.environ.get("SPEED", "0.1"))       # 0.1 = 10% of max speed
BASE = "base_link"
CAMERA = os.environ.get("CAMERA_FRAME", "camera_rgb_optical_frame")
# Safety box in base_link (meters). Refuse targets outside it.
MIN_REACH, MAX_REACH = 0.25, 0.65   # very close to the base the arm folds into itself
# The table is ~5 cm BELOW base_link (base sits on a 52.4 mm plate), and tool_frame is
# ~32 mm above the fingertips. tool_frame at z=0.0 -> fingertips ~2 cm above the table.
MIN_Z = float(os.environ.get("MIN_Z", "-0.02"))


def to_frame(buf, pt, frame):
    """Transform a PointStamped using the LATEST tf (so a re-tuned calibration is used)."""
    pt = PointStamped(header=pt.header, point=pt.point)
    pt.header.stamp = rospy.Time(0)
    return buf.transform(pt, frame, rospy.Duration(1.0))


def hover(arm, p):
    """Check the safety box, ask, and move tool_frame to HOVER_M above base_link point p."""
    reach = math.hypot(p.x, p.y)
    target_z = p.z + HOVER_M
    print(f"Point in {BASE}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}  (reach {reach:.2f} m)")
    if not (MIN_REACH <= reach <= MAX_REACH):
        print(f"Reach {reach:.2f} m is outside {MIN_REACH}-{MAX_REACH} m - pick another spot.")
        return
    if target_z < MIN_Z:
        print(f"Target z={target_z:.3f} is below the floor MIN_Z={MIN_Z} - use a bigger HOVER_M.")
        return
    if input(f"Move gripper to z={target_z:.3f} above it? [y/N] (gamepad RB = E-stop) ").strip().lower() != "y":
        return

    pose = arm.get_current_pose().pose          # keep current orientation
    pose.position.x, pose.position.y, pose.position.z = p.x, p.y, target_z
    arm.set_pose_target(pose)
    ok = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not ok:
        print("Planning/execution failed for that point.")
        return
    now = arm.get_current_pose().pose.position
    print(f"Gripper now at x={now.x:.3f} y={now.y:.3f} z={now.z:.3f}. "
          "Measure the sideways gap to the spot.")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("hover_test")

    arm = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
    arm.set_max_velocity_scaling_factor(SPEED)
    arm.set_max_acceleration_scaling_factor(SPEED)

    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference or it stops listening

    print(f"End effector: {arm.get_end_effector_link()}  |  hover {HOVER_M} m  |  speed {SPEED:.0%}")
    saved = None   # last clicked point, stored in the CAMERA frame

    while not rospy.is_shutdown():
        if saved is None:
            print("\nClick a point in RViz with 'Publish Point'. Ctrl+C to quit.")
            clicked = rospy.wait_for_message("/clicked_point", PointStamped)
            try:
                # RViz gives the point in its Fixed Frame (base_link). Store it relative to
                # the camera, so later calibration tweaks move where it lands in base_link.
                saved = to_frame(buf, clicked, CAMERA)
            except Exception as e:
                print(f"Couldn't convert to {CAMERA}: {e}\n(Is publish_calibration.py running?)")
                continue
        else:
            cmd = input("\n[Enter] same point again (uses current calibration) | "
                        "c = click a new point | q = quit: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "c":
                saved = None
                continue

        try:
            p = to_frame(buf, saved, BASE).point
        except Exception as e:
            print(f"Couldn't convert to {BASE}: {e}\n(Is publish_calibration.py running?)")
            continue
        hover(arm, p)


if __name__ == "__main__":
    main()