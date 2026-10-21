#!/usr/bin/env python3
"""Smallest possible real-arm test: rotate the base joint 0.3 rad (~17 deg) and back.

Needs start_driver.sh running in another terminal. Then, inside the container:
    cd /root/kinova_ws/src/shawn_scripts
    python3 one_move.py
"""
import os
import sys
# Kinova launches everything under /my_gen3_lite; set it here so you don't have to type it.
os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import rospy
import moveit_commander

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node("one_move")

arm = moveit_commander.MoveGroupCommander("arm", ns=rospy.get_namespace())
arm.set_max_velocity_scaling_factor(0.1)       # slow
arm.set_max_acceleration_scaling_factor(0.1)

joints = arm.get_current_joint_values()
print("Current joints (rad):", [round(j, 3) for j in joints])

input("Press Enter to rotate the base +0.3 rad (gamepad RB = E-stop)...")
joints[0] += 0.3
arm.go(joints, wait=True)
arm.stop()

input("Press Enter to rotate back...")
joints[0] -= 0.3
arm.go(joints, wait=True)
arm.stop()

print("Done. Joints now:", [round(j, 3) for j in arm.get_current_joint_values()])
moveit_commander.roscpp_shutdown()