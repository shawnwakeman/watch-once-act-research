#!/usr/bin/env python3
"""Click-driven pick and place of a cube of known size.

  1. Click the CENTER OF THE CUBE'S TOP FACE in RViz ("Publish Point").
  2. Click a spot ON THE TABLE where it should go.
  3. Type y. The arm: opens -> moves above the cube -> descends straight down ->
     closes -> lifts -> moves above the place spot -> descends -> opens -> lifts.

The gripper always points straight down with a FIXED twist about vertical (YAW_DEG,
default 180), so line the cube's faces up with the fingers at that angle.

E-stop (gamepad RB) works like hover_simple: the move in progress fails, the sequence
stops at that step, and the script goes back to waiting for clicks.

Needs running: start_driver.sh, start_camera.sh, publish_calibration.py, RViz.
Usage (inside the container):
    CUBE_M=0.05 python3 pick_place.py
Settings (env vars):
    CUBE_M   cube edge length, m (default 0.05)
    TABLE_Z  table height in base_link, m (default -0.052: base sits on the 52.4 mm plate)
    SPEED    0.1 = 10% (default)
    YAW_DEG  gripper twist about vertical, degrees (default 180; 0 also works)
    CLOSE    gripper close amount 0..1 (default 0.9; 0 = open, 1 = fully closed)
    GRIP_FLIP  set to 1 if open/close come out backwards
The gripper is driven straight through the driver's gripper action
(control_msgs/GripperCommand), not through MoveIt's gripper group.
"""
import math
import os
import sys
import time

os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import actionlib
import numpy as np
import rospy
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
import moveit_commander
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from geometry_msgs.msg import PointStamped, PoseStamped
from tf.transformations import quaternion_from_matrix

CUBE = float(os.environ.get("CUBE_M", "0.04"))
TABLE_Z = float(os.environ.get("TABLE_Z", "-0.052"))
SPEED = min(float(os.environ.get("SPEED", "1.0")), 1.0)  
CLOSE = float(os.environ.get("CLOSE", "0.8"))
YAW_DEG = float(os.environ.get("YAW_DEG", "180"))
GRIP_FLIP = os.environ.get("GRIP_FLIP", "0") == "1"
BASE = "base_link"
TIP = 0.032           # tool_frame is ~32 mm above the fingertips (tip check)
APPROACH = 0.10       # pre-grasp / pre-place height above the target, m
CLEAR = 0.004         # never put the fingertips closer than this to the table
GRIP_DEPTH = min(0.6 * CUBE, CUBE - 0.008)   # how far down the cube the fingertips go
MIN_REACH, MAX_REACH = 0.15, 0.65


def down_with_yaw(yaw_deg):
    """Tool z-axis pointing straight down, tool x-axis at yaw_deg about vertical.
    yaw 0 -> tool x along base +x;  yaw 180 -> tool x along base -x."""
    a = math.radians(yaw_deg)
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([math.cos(a), math.sin(a), 0.0])
    M = np.eye(4)
    M[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return quaternion_from_matrix(M)


class Robot:
    def __init__(self):
        ns = rospy.get_namespace()
        self.arm = moveit_commander.MoveGroupCommander("arm", ns=ns)
        self.arm.set_max_velocity_scaling_factor(SPEED)
        self.arm.set_max_acceleration_scaling_factor(SPEED)
        # Gripper: read limits/position via the robot model, command via the driver's action.
        self.robot = moveit_commander.RobotCommander("robot_description")
        names = rospy.get_param(ns + "gripper_joint_names", [])
        if not names:
            sys.exit(f"No {ns}gripper_joint_names param - was the driver started with gripper:=gen3_lite_2f?")
        self.grip_joint = self.robot.get_joint(names[0])
        lo, hi = self.grip_joint.min_bound(), self.grip_joint.max_bound()
        # On this Gen3 Lite the HIGH end reads open (0.96) and the LOW end closed (-0.09).
        self.grip_open, self.grip_closed = (lo, hi) if GRIP_FLIP else (hi, lo)
        topics = [t for t, _ in rospy.get_published_topics() if t.endswith("gripper_cmd/status")]
        if not topics:
            sys.exit("No gripper action found (no .../gripper_cmd/status topic) - is the driver running?")
        action = topics[0][:-len("/status")]
        self.grip_client = actionlib.SimpleActionClient(action, GripperCommandAction)
        if not self.grip_client.wait_for_server(rospy.Duration(5)):
            sys.exit(f"Gripper action {action} not responding")
        print(f"Gripper action {action} | open={self.grip_open:.2f} closed={self.grip_closed:.2f} "
              f"| now {self.grip_joint.value():.2f}")
        self.quat = down_with_yaw(YAW_DEG)

        # Table as a collision box so planned moves never swing through it.
        scene = moveit_commander.PlanningSceneInterface(ns=ns, synchronous=True)
        box = PoseStamped()
        box.header.frame_id = BASE
        box.pose.position.z = TABLE_Z - 0.005 - 0.01     # top 5 mm below the real table
        box.pose.orientation.w = 1.0
        scene.add_box("table", box, size=(2.0, 2.0, 0.02))

    def grip_pos(self, value):
        """0 = open, 1 = closed -> absolute joint position, kept 2% inside the limits."""
        rng = self.grip_closed - self.grip_open
        target = self.grip_open + value * rng
        lo, hi = sorted((self.grip_open, self.grip_closed))
        m = 0.02 * (hi - lo)
        return min(max(target, lo + m), hi - m)

    def send_grip(self, value):
        goal = GripperCommandGoal()
        goal.command.position = self.grip_pos(value)
        goal.command.max_effort = 100.0
        self.grip_client.send_goal(goal)
        # The fingers stall on the cube / stop short of the limit, so the action may never
        # report success - wait a bounded time, then judge by the measured position.
        self.grip_client.wait_for_result(rospy.Duration(3.0))
        time.sleep(0.3)
        return self.grip_joint.value()

    def open_gripper(self):
        pos = self.send_grip(0.0)
        span = abs(self.grip_closed - self.grip_open)
        ok = abs(pos - self.grip_open) < 0.15 * span
        print(f"  gripper at {pos:.2f} (open = {self.grip_open:.2f}) -> {'ok' if ok else 'NOT open'}")
        return ok

    def close_gripper(self):
        """Close and assume the cube is in the gripper (no grasp check)."""
        pos = self.send_grip(CLOSE)
        print(f"  gripper at {pos:.2f} - assuming it has the cube")
        return True

    def pose_at(self, x, y, z):
        p = self.arm.get_current_pose().pose
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = self.quat
        return p

    def move_to(self, x, y, z):
        """Planned move (free path) - used for the big moves between targets."""
        self.arm.set_pose_target(self.pose_at(x, y, z))
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return ok

    def go_joints(self, joints):
        """Planned move back to a saved set of joint angles."""
        self.arm.set_joint_value_target(joints)
        ok = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return ok

    def straight_to(self, z):
        """Straight-line vertical move (for descending onto / lifting off the cube).

        The arm validates every trajectory and rejects ones whose waypoints need too much
        acceleration ("Invalid Acceleration ... Segment (N)"). A straight line made of many
        tiny IK steps can contain such spikes, so: use coarser 1 cm steps, smooth timing
        (time-optimal parameterization) with gentler acceleration, and if the arm still
        rejects it, fall back to a normal planned move to the same pose."""
        cur = self.arm.get_current_pose().pose.position
        target = self.pose_at(cur.x, cur.y, z)
        try:    # newer MoveIt Noetic: (waypoints, eef_step, avoid_collisions)
            plan, frac = self.arm.compute_cartesian_path([target], 0.01, True)
        except Exception:   # older MoveIt: (waypoints, eef_step, jump_threshold)
            plan, frac = self.arm.compute_cartesian_path([target], 0.01, 0.0)
        print(f"  straight-line path to z={z:.3f}: {frac:.0%} planned")
        ok = False
        if frac >= 0.95:
            acc = 0.5 * SPEED
            try:
                plan = self.arm.retime_trajectory(self.arm.get_current_state(), plan,
                                                  velocity_scaling_factor=SPEED,
                                                  acceleration_scaling_factor=acc,
                                                  algorithm="time_optimal_trajectory_generation")
            except Exception:
                plan = self.arm.retime_trajectory(self.arm.get_current_state(), plan,
                                                  velocity_scaling_factor=SPEED,
                                                  acceleration_scaling_factor=acc)
            ok = self.arm.execute(plan, wait=True)
            self.arm.stop()
        if not ok:
            print("  straight-line move rejected/failed - trying a normal planned move instead")
            self.arm.set_pose_target(target)
            ok = self.arm.go(wait=True)
            self.arm.stop()
            self.arm.clear_pose_targets()
        return ok


def get_click(buf, what):
    print(f"\nClick {what} in RViz ...")
    while not rospy.is_shutdown():
        c = rospy.wait_for_message("/clicked_point", PointStamped)
        c.header.stamp = rospy.Time(0)
        try:
            p = buf.transform(c, BASE, rospy.Duration(1.0)).point
        except Exception as e:
            print(f"  couldn't convert to {BASE}: {e} (is publish_calibration.py running?)")
            continue
        reach = math.hypot(p.x, p.y)
        print(f"  x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}  (reach {reach:.2f} m, "
              f"{100 * (p.z - TABLE_Z):.1f} cm above table)")
        if not MIN_REACH <= reach <= MAX_REACH:
            print(f"  reach outside {MIN_REACH}-{MAX_REACH} m - click again")
            continue
        return p


def run(robot, cube, place):
    # Use the table height the camera actually sees at each spot (the table/cloud can be
    # off by several mm from TABLE_Z), not the global TABLE_Z.
    pick_table = cube.z - CUBE
    place_table = place.z
    grasp_tips = max(cube.z - GRIP_DEPTH, pick_table + CLEAR)
    grasp_z = grasp_tips + TIP
    # fingertips sit (CUBE - how far they went down from the top) above the cube's bottom
    place_tips = place_table + 0.010 + (CUBE - (cube.z - grasp_tips))  # bottom ~1 cm above table (camera z is +-5-9 mm)
    place_z = place_tips + TIP
    hi_pick = cube.z + APPROACH + TIP
    hi_place = place_table + CUBE + APPROACH + TIP

    start = robot.arm.get_current_joint_values()   # where the arm was before this run

    steps = [
        ("open gripper",           lambda: robot.open_gripper()),
        ("move above cube",        lambda: robot.move_to(cube.x, cube.y, hi_pick)),
        ("open gripper",           lambda: robot.open_gripper()),
        ("descend",                lambda: robot.straight_to(grasp_z)),
        ("close gripper",          lambda: robot.close_gripper()),
        ("lift",                   lambda: robot.straight_to(hi_pick)),
        ("move above place spot",  lambda: robot.move_to(place.x, place.y, hi_place)),
        ("descend",                lambda: robot.straight_to(place_z)),
        ("open gripper",           lambda: robot.open_gripper()),
        ("lift",                   lambda: robot.straight_to(hi_place)),
        ("return to start pose",   lambda: robot.go_joints(start)),
    ]
    for name, fn in steps:
        print(f"  -> {name}")
        if not fn():
            print(f"  FAILED at '{name}' - stopping here (arm left where it is).")
            return
    print("  done.")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("pick_place")
    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference
    robot = Robot()
    print(f"Cube {CUBE * 100:.1f} cm | yaw {YAW_DEG:.0f} deg | table z={TABLE_Z} | grip depth {GRIP_DEPTH * 100:.1f} cm "
          f"| speed {SPEED:.0%}. Ctrl+C to quit.")

    while not rospy.is_shutdown():
        cube = get_click(buf, "the CENTER OF THE CUBE'S TOP FACE")
        place = get_click(buf, "a spot ON THE TABLE to place it")
        if abs(place.z - TABLE_Z) > 0.015:
            print(f"  that spot is {100 * (place.z - TABLE_Z):.1f} cm off the table height - "
                  "click bare table (or fix TABLE_Z)")
            continue
        if input("Run pick and place? [y/N] (gamepad RB = E-stop) ").strip().lower() == "y":
            run(robot, cube, place)


if __name__ == "__main__":
    main()