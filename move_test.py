#!/usr/bin/env python3
"""Gen3 Lite demo sequence via MoveIt (works in Gazebo and on the real arm).

Needs start_driver.sh (real arm) or the Gazebo sim running. Then, inside the container:
    cd /root/kinova_ws/src/shawn_scripts
    python3 move_test.py

Speed defaults to 10% for the real arm. In sim you can go faster:
    SPEED=0.3 python3 move_test.py
"""
import os
import sys
# Kinova launches everything under /my_gen3_lite; set it here so you don't have to type it.
os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import rospy
import moveit_commander
import geometry_msgs.msg

SPEED = float(os.environ.get("SPEED", "0.1"))


def go_named(arm, name):
    if name in arm.get_named_targets():
        print(f"-> {name}")
        arm.set_named_target(name)
        arm.go(wait=True)
        arm.stop()


def move_ee(arm, dx=0.0, dy=0.0, dz=0.0):
    pose = arm.get_current_pose().pose
    target = geometry_msgs.msg.Pose()
    target.position.x = pose.position.x + dx
    target.position.y = pose.position.y + dy
    target.position.z = pose.position.z + dz
    target.orientation = pose.orientation
    print(f"-> move end effector by ({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) m")
    arm.set_pose_target(target)
    ok = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    if not ok:
        print("   (planning failed for that move, skipping)")


def set_gripper(robot, fraction):
    """fraction: 0.0 = open, 1.0 = closed (same approach as Kinova's MoveIt example)."""
    try:
        joint = robot.get_joint("right_finger_bottom_joint")
        lo, hi = joint.min_bound(), joint.max_bound()
        print(f"-> gripper {'close' if fraction > 0.5 else 'open'}")
        joint.move(lo + fraction * (hi - lo), True)
    except Exception as e:
        print(f"   (gripper move skipped: {e})")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("move_test")

    robot = moveit_commander.RobotCommander("robot_description")
    arm = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
    arm.set_max_velocity_scaling_factor(SPEED)
    arm.set_max_acceleration_scaling_factor(SPEED)

    print("Speed scaling: ", SPEED)
    print("Planning frame:", arm.get_planning_frame())
    print("End effector:  ", arm.get_end_effector_link())
    print("Named poses:   ", arm.get_named_targets())
    input("Clear the area around the arm, then press Enter (gamepad RB = E-stop)...")

    go_named(arm, "home")

    # Rotate base out and back
    joints = arm.get_current_joint_values()
    start_base = joints[0]
    for offset in (0.6, -0.6, 0.0):
        joints[0] = start_base + offset
        print(f"-> base joint to {joints[0]:+.2f} rad")
        arm.go(joints, wait=True)
        arm.stop()

    # Fake "pick": lower, close gripper, lift, open
    set_gripper(robot, 0.0)
    move_ee(arm, dz=-0.08)
    set_gripper(robot, 0.8)
    move_ee(arm, dz=+0.08)
    move_ee(arm, dy=+0.08)
    set_gripper(robot, 0.0)
    move_ee(arm, dy=-0.08)

    go_named(arm, "retract")
    go_named(arm, "home")

    print("Done. Final position:", arm.get_current_pose().pose.position)
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()