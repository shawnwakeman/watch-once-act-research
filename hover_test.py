#!/usr/bin/env python3
"""Simple hover: click a point in RViz ("Publish Point"), the gripper moves above it.
No prompts, no repeat mode - just click, move, click, move.

The hover height is the gap between the FINGERTIPS and the clicked point: h 0.04 puts
the fingertips 4 cm above it (a 4 cm cube fits underneath). The script adds the 32 mm
between the fingertips and tool_frame (TIP) itself.

Change settings while it runs by typing in this terminal (then Enter):
    h 0.04      hover height: fingertips this far above the point, m
    s 0.2       speed (0.1 = 10%, max 0.5 here)
    show        print current settings
New values apply on the next click.

The gripper is turned to point STRAIGHT DOWN on every move (keeping its current
twist about vertical), so you don't need to pose the arm first.

Needs running: start_driver.sh, start_camera.sh, publish_calibration.py, RViz.
Usage (inside the container):
    python3 hover_simple.py
    HOVER_M=0.04 python3 hover_simple.py     # fingertips above the point (default 0.10)
    SPEED=0.2 python3 hover_simple.py        # default 0.1 = 10%
"""
import math
import os
import sys
import threading

os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import numpy as np
import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_matrix, quaternion_from_matrix

SET = {"hover": float(os.environ.get("HOVER_M", "0.10")),
       "speed": float(os.environ.get("SPEED", "0.7"))}
MAX_SPEED = 1
BASE = "base_link"
MIN_REACH, MAX_REACH = 0.15, 0.65
TIP = 0.032      # tool_frame is ~32 mm above the fingertips (same as pick_place.py)
MIN_Z = 0.0      # tool_frame never below z=0 -> fingertips never below ~2 cm above the table


def straight_down(q):
    """Orientation with the tool z-axis pointing down, keeping the current twist."""
    R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    z = np.array([0.0, 0.0, -1.0])
    x = R[:, 0] - R[:, 0].dot(z) * z          # current tool x, flattened to horizontal
    if np.linalg.norm(x) < 1e-3:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, :3] = np.column_stack([x, y, z])
    return quaternion_from_matrix(M)


def read_commands():
    """Background thread: lets you change hover/speed by typing while it waits for clicks."""
    while True:
        try:
            cmd = input().strip().lower().split()
        except EOFError:
            return
        if not cmd:
            continue
        if cmd[0] == "show":
            print(f"  hover: fingertips {SET['hover']} m above the point | speed {SET['speed']:.0%}")
            continue
        if cmd[0] in ("h", "s") and len(cmd) == 2:
            try:
                v = float(cmd[1])
            except ValueError:
                print("  use e.g. 'h 0.04' or 's 0.2'"); continue
            if cmd[0] == "h":
                SET["hover"] = v
                print(f"  hover -> fingertips {v} m above the point (next click)")
            elif 0 < v <= MAX_SPEED:
                SET["speed"] = v
                print(f"  speed -> {v:.0%} (next click)")
            else:
                print(f"  speed must be between 0 and {MAX_SPEED}")
            continue
        print("  commands: 'h 0.04', 's 0.2', 'show'")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("hover_simple")
    arm = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference or it stops listening

    threading.Thread(target=read_commands, daemon=True).start()
    print(f"hover: fingertips {SET['hover']} m above the point | speed {SET['speed']:.0%} | "
          "click points in RViz, Ctrl+C to quit")
    print("type 'h 0.04' / 's 0.2' / 'show' + Enter to change settings")
    while not rospy.is_shutdown():
        clicked = rospy.wait_for_message("/clicked_point", PointStamped)
        clicked.header.stamp = rospy.Time(0)
        try:
            p = buf.transform(clicked, BASE, rospy.Duration(1.0)).point
        except Exception as e:
            print(f"Couldn't convert to {BASE}: {e} (is publish_calibration.py running?)")
            continue

        reach = math.hypot(p.x, p.y)
        hover, speed = SET["hover"], SET["speed"]
        z = p.z + hover + TIP              # tool_frame height; fingertips end up `hover` above the point
        print(f"\nClicked x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} (reach {reach:.2f} m)")
        if not MIN_REACH <= reach <= MAX_REACH:
            print(f"  skipped: reach outside {MIN_REACH}-{MAX_REACH} m"); continue
        if z < MIN_Z:
            print(f"  skipped: fingertips would be {100 * (z - TIP - p.z):.1f} cm above the point but tool_frame "
                  f"z={z:.3f} is below the {MIN_Z} floor - use a bigger hover"); continue

        arm.set_max_velocity_scaling_factor(speed)
        arm.set_max_acceleration_scaling_factor(speed)
        pose = arm.get_current_pose().pose
        pose.position.x, pose.position.y, pose.position.z = p.x, p.y, z
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = straight_down(pose.orientation)
        arm.set_pose_target(pose)
        ok = arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()
        if not ok:
            print("  move failed (no plan / out of reach for that orientation)"); continue
        now = arm.get_current_pose().pose.position
        print(f"  fingertips at z={now.z - TIP:.3f} = {100 * (now.z - TIP - p.z):.1f} cm above the point "
              f"(tool_frame x={now.x:.3f} y={now.y:.3f} z={now.z:.3f}, speed {speed:.0%})")


if __name__ == "__main__":
    main()