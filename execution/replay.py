#!/usr/bin/env python3
"""Replay a demo on the arm: the object-centric trajectory from the pipeline, applied to
wherever the objects are NOW (seen by the robot's Xtion).

  pick_place  move the cube by the same displacement as in the demo
  stack2      place the cube where it ended relative to the grey block in the demo
  bowl_drop   drop the cube where it was released relative to the box in the demo
  slide_path  push the cube along the demo's path (split into straight pushes), looking
              again after each push and correcting toward the path

The demo comes from lift.py (Mac): copy each run's perception/xtion/trajectory.yaml and
trajectory.npz to the lab PC, e.g. shawn_scripts/demos_l0/pick_place/run01/.

Finding the objects live (held constant across all video-quality levels, so it doesn't
bias the study):
  cube    pink-color segmentation + depth (automatic), or --cube click
  target  RViz click on the grey block's top / the box's center (--target click)

Usage (inside the container; driver, camera, publish_calibration.py and RViz running):
  python3 replay.py pick_place demos_l0/pick_place/run01 --hover-only   # check first!
  python3 replay.py pick_place demos_l0/pick_place/run01
  python3 replay.py stack2     demos_l0/stack2/run03
  python3 replay.py bowl_drop  demos_l0/bowl_drop/run02
  python3 replay.py slide_path demos_l0/slide_path/run01
Settings (env vars, plus all of pick_place.py's: CUBE_M, SPEED, YAW_DEG, ...):
  PINK_HSV      "140,70,70,179,255,255"  HSV low/high for the cube (OpenCV: H 0-179)
  DROP_ABOVE_M  0.06   bowl_drop: cube bottom this far above the box's clicked point at release
  PUSH_LOW_M    0.012  slide: fingertips this far above the table while pushing
  PUSH_YAW_OFF  90     slide: gripper twist relative to the push direction, degrees
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("ROS_NAMESPACE", "my_gen3_lite")

import cv2
import numpy as np
import rospy
import moveit_commander
import tf2_ros
import yaml
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick_place as pp                                       # noqa: E402  (motion + gripper code)

PINK = [int(v) for v in os.environ.get("PINK_HSV", "140,70,70,179,255,255").split(",")]
DROP_ABOVE = float(os.environ.get("DROP_ABOVE_M", "0.06"))
PUSH_LOW = float(os.environ.get("PUSH_LOW_M", "0.012"))
PUSH_YAW_OFF = float(os.environ.get("PUSH_YAW_OFF", "90"))
RGB_T, DEPTH_T, INFO_T = "/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw", "/camera/rgb/camera_info"


class P:                       # a tiny point type, compatible with pick_place.run()
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


# ------------------------------------------------------------------ live camera ---
class Camera:
    """One snapshot of color + depth + intrinsics, and the camera's pose in base_link."""

    def __init__(self, buf):
        self.buf = buf

    def snap(self):
        rgb = rospy.wait_for_message(RGB_T, Image, timeout=5)
        dep = rospy.wait_for_message(DEPTH_T, Image, timeout=5)
        info = rospy.wait_for_message(INFO_T, CameraInfo, timeout=5)
        a = np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.width, -1)
        self.bgr = a[:, :, ::-1].copy() if rgb.encoding.lower() == "rgb8" else a.copy()
        d = np.frombuffer(dep.data, np.uint16 if dep.encoding.upper() == "16UC1" else np.float32)
        d = d.reshape(dep.height, dep.width)
        self.depth = d.astype(np.float32) / 1000.0 if d.dtype == np.uint16 else np.nan_to_num(d)   # meters
        K = np.array(info.K).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        tr = self.buf.lookup_transform(pp.BASE, rgb.header.frame_id, rospy.Time(0), rospy.Duration(2))
        t, q = tr.transform.translation, tr.transform.rotation
        self.T = quaternion_matrix([q.x, q.y, q.z, q.w])
        self.T[:3, 3] = [t.x, t.y, t.z]
        return self

    def to_base(self, us, vs):
        """Pixels -> base_link points (nan where no depth)."""
        Z = self.depth[vs, us]
        Z = np.where(Z > 0, Z, np.nan)
        pc = np.stack([(us - self.cx) * Z / self.fx, (vs - self.cy) * Z / self.fy, Z], 1)
        return (self.T[:3, :3] @ pc.T).T + self.T[:3, 3]

    def surface_z_at(self, x, y, z_guess, r=4):
        """Height of whatever surface the camera sees at base (x, y)."""
        inv = np.linalg.inv(self.T)
        pc = inv[:3, :3] @ np.array([x, y, z_guess]) + inv[:3, 3]
        u, v = int(round(self.fx * pc[0] / pc[2] + self.cx)), int(round(self.fy * pc[1] / pc[2] + self.cy))
        h, w = self.depth.shape
        if not (r <= u < w - r and r <= v < h - r):
            return None
        vs, us = np.mgrid[v - r:v + r + 1, u - r:u + r + 1]
        pts = self.to_base(us.ravel(), vs.ravel())
        pts = pts[~np.isnan(pts[:, 2])]
        return float(np.median(pts[:, 2])) if len(pts) else None

    def find_pink_cube(self):
        """Top-face center of the pink cube (largest pink blob) in base_link."""
        hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array(PINK[:3]), np.array(PINK[3:]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = cv2.erode((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))
        vs, us = np.nonzero(blob)
        pts = self.to_base(us, vs)
        pts = pts[~np.isnan(pts[:, 2])]
        if len(pts) < 30:
            return None
        top = np.percentile(pts[:, 2], 90)
        sel = pts[pts[:, 2] > top - 0.008]
        cv2.imwrite("/tmp/replay_cube_mask.png", np.hstack([self.bgr, cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR)]))
        return P(np.median(sel[:, 0]), np.median(sel[:, 1]), top)


def locate_cube(cam, how, buf):
    if how == "color":
        c = cam.snap().find_pink_cube()
        if c is None:
            print("  !! no pink cube found (tune PINK_HSV, see /tmp/replay_cube_mask.png) - click it instead")
        else:
            print(f"  cube top (color+depth): {c}   (mask: /tmp/replay_cube_mask.png)")
            return c
    p = pp.get_click(buf, "the CENTER OF THE CUBE'S TOP FACE")
    return P(p.x, p.y, p.z)


# --------------------------------------------------------------------- the demo ---
def load_demo(path):
    d = Path(path)
    summ = yaml.safe_load((d / "trajectory.yaml").read_text())
    npz = d / "trajectory.npz"
    xyz = np.load(npz)["T_base"][:, :3, 3] if npz.exists() else None
    return summ, xyz


def simplify(points, tol):
    """Douglas-Peucker: the corners of a path (2D)."""
    if len(points) < 3:
        return list(range(len(points)))
    a, b = points[0], points[-1]
    ab = b - a
    L = np.linalg.norm(ab) + 1e-12
    d = np.abs(ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])) / L
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [0, len(points) - 1]
    left = simplify(points[:i + 1], tol)
    right = simplify(points[i:], tol)
    return left[:-1] + [i + j for j in right]


# --------------------------------------------------------------------- motions ---
def straight_xyz(robot, x, y, z):
    """Straight line to (x, y, z) keeping the current orientation (same retiming and
    planned-move fallback as pick_place's vertical moves)."""
    target = robot.pose_at(x, y, z)
    try:
        plan, frac = robot.arm.compute_cartesian_path([target], 0.01, True)
    except Exception:
        plan, frac = robot.arm.compute_cartesian_path([target], 0.01, 0.0)
    ok = False
    if frac >= 0.95:
        try:
            plan = robot.arm.retime_trajectory(robot.arm.get_current_state(), plan,
                                               velocity_scaling_factor=pp.SPEED,
                                               acceleration_scaling_factor=0.5 * pp.SPEED,
                                               algorithm="time_optimal_trajectory_generation")
        except Exception:
            plan = robot.arm.retime_trajectory(robot.arm.get_current_state(), plan,
                                               velocity_scaling_factor=pp.SPEED,
                                               acceleration_scaling_factor=0.5 * pp.SPEED)
        ok = robot.arm.execute(plan, wait=True)
        robot.arm.stop()
    if not ok:
        print("  straight line not possible - planned move instead")
        robot.arm.set_pose_target(target)
        ok = robot.arm.go(wait=True)
        robot.arm.stop()
        robot.arm.clear_pose_targets()
    return ok


def check_reach(*pts):
    for p in pts:
        r = math.hypot(p.x, p.y)
        if not pp.MIN_REACH <= r <= pp.MAX_REACH:
            print(f"  !! {p} is {r:.2f} m from the base - outside {pp.MIN_REACH}-{pp.MAX_REACH} m")
            return False
    return True


def drop(robot, cube, spot, release_z_bottom):
    """Pick the cube, carry it above spot, open (no set-down)."""
    grasp_tips = max(cube.z - pp.GRIP_DEPTH, cube.z - pp.CUBE + pp.CLEAR)
    hold = cube.z - grasp_tips                       # fingertips this far below the cube's top
    rel_tool = release_z_bottom + pp.CUBE - hold + pp.TIP
    hi = max(rel_tool, cube.z + pp.TIP) + pp.APPROACH
    start = robot.arm.get_current_joint_values()
    steps = [("open", robot.open_gripper), ("above cube", lambda: robot.move_to(cube.x, cube.y, cube.z + pp.APPROACH + pp.TIP)),
             ("open", robot.open_gripper), ("descend", lambda: robot.straight_to(grasp_tips + pp.TIP)),
             ("close", robot.close_gripper), ("lift", lambda: robot.straight_to(cube.z + pp.APPROACH + pp.TIP)),
             ("above drop spot", lambda: robot.move_to(spot.x, spot.y, hi)),
             ("lower to release height", lambda: robot.straight_to(rel_tool)),
             ("release", robot.open_gripper), ("lift", lambda: robot.straight_to(hi)),
             ("return", lambda: robot.go_joints(start))]
    for name, fn in steps:
        print(f"  -> {name}")
        if not fn():
            print(f"  FAILED at '{name}' - stopping (arm left where it is).")
            return False
    return True


def push_path(robot, cam, cube, path_xy, buf, how):
    """Push the cube along path_xy (cube-center waypoints, base frame). After each straight
    push, look again and aim the next push from where the cube really is."""
    half = pp.CUBE / 2
    standoff = half + 0.03                           # where the fingers start, behind the cube
    table = cube.z - pp.CUBE
    low = table + PUSH_LOW + pp.TIP                  # tool height while pushing
    safe = cube.z + pp.APPROACH + pp.TIP
    start = robot.arm.get_current_joint_values()
    robot.close_gripper()
    cur = np.array([cube.x, cube.y])
    for k, goal in enumerate(path_xy[1:], 1):
        d = goal - cur
        L = float(np.linalg.norm(d))
        if L < 0.01:
            continue
        u = d / L
        behind = cur - u * standoff
        stop = goal - u * (half + 0.003)             # finger stops at the cube's back face at the goal
        yaw = math.degrees(math.atan2(u[1], u[0])) + PUSH_YAW_OFF
        robot.quat = pp.down_with_yaw(yaw)
        print(f"  push {k}/{len(path_xy) - 1}: {L * 100:.1f} cm toward ({goal[0]:.3f}, {goal[1]:.3f})")
        for name, fn in [("above start", lambda: robot.move_to(behind[0], behind[1], safe)),
                         ("lower", lambda: robot.straight_to(low)),
                         ("push", lambda: straight_xyz(robot, stop[0], stop[1], low)),
                         ("back off", lambda: straight_xyz(robot, stop[0] - u[0] * 0.02, stop[1] - u[1] * 0.02, low)),
                         ("raise", lambda: robot.straight_to(safe))]:
            if not fn():
                print(f"  FAILED at '{name}' - stopping (arm left where it is).")
                return False
        if k < len(path_xy) - 1:                     # feedback: where did it really go?
            seen = locate_cube(cam, how, buf)
            print(f"    cube now at ({seen.x:.3f}, {seen.y:.3f}); planned ({goal[0]:.3f}, {goal[1]:.3f}) "
                  f"-> off by {100 * math.hypot(seen.x - goal[0], seen.y - goal[1]):.1f} cm")
            cur = np.array([seen.x, seen.y])
    robot.quat = pp.down_with_yaw(pp.YAW_DEG)
    return robot.go_joints(start)


# ------------------------------------------------------------------------ main ---
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["pick_place", "stack2", "bowl_drop", "slide_path"])
    ap.add_argument("demo", help="folder with the run's trajectory.yaml (+ trajectory.npz)")
    ap.add_argument("--cube", choices=["color", "click"], default="color")
    ap.add_argument("--hover-only", action="store_true", help="just hover above the computed goal(s)")
    a = ap.parse_args()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("replay")
    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference
    rospy.sleep(1.0)
    robot = pp.Robot()
    cam = Camera(buf)
    summ, xyz = load_demo(a.demo)
    print(f"Demo {a.demo}: displacement {summ['displacement_m']} m, end rotation {summ['end_rotation_deg']} deg")

    cube = locate_cube(cam, a.cube, buf)

    if a.task == "pick_place":
        dx, dy = summ["displacement_m"][:2]
        gx, gy = cube.x + dx, cube.y + dy
        gz = cam.surface_z_at(gx, gy, cube.z - pp.CUBE)
        gz = gz if gz is not None and abs(gz - (cube.z - pp.CUBE)) < 0.03 else cube.z - pp.CUBE
        goal = P(gx, gy, gz)                         # the table surface where it goes
        print(f"  goal (table under the cube's new spot): {goal}  = cube + demo displacement")
    elif a.task in ("stack2", "bowl_drop"):
        if "end_minus_target_m" not in summ:
            raise SystemExit("This demo has no target (run perceive.py with --target, then lift.py).")
        what = "the CENTER OF THE GREY BLOCK'S TOP" if a.task == "stack2" else "the CENTER OF THE BOX (inside bottom)"
        tgt = pp.get_click(buf, what)
        ex, ey, _ = summ["end_minus_target_m"]
        goal = P(tgt.x + ex, tgt.y + ey, tgt.z)
        print(f"  target {P(tgt.x, tgt.y, tgt.z)} + demo offset ({ex * 100:.1f}, {ey * 100:.1f}) cm -> goal {goal}")
    else:  # slide_path
        if xyz is None:
            raise SystemExit("slide_path needs trajectory.npz next to trajectory.yaml")
        rel = xyz[:, :2] - xyz[0, :2]
        idx = simplify(rel, 0.01)
        path = np.array([cube.x, cube.y]) + rel[idx]
        goal = P(path[-1, 0], path[-1, 1], cube.z)
        print(f"  path: {len(idx) - 1} straight pushes, corners at "
              + ", ".join(f"({p[0]:.3f}, {p[1]:.3f})" for p in path))

    if not check_reach(cube, goal):
        return
    if a.hover_only:
        print("  hover check: above the cube, then above the goal")
        robot.move_to(cube.x, cube.y, cube.z + 0.05 + pp.TIP)
        input("  above the cube - Enter for the goal ")
        robot.move_to(goal.x, goal.y, goal.z + pp.CUBE + 0.05 + pp.TIP)
        print("  above the goal. Check by eye, then run without --hover-only.")
        return
    if input(f"Run {a.task}? [y/N] (gamepad RB = E-stop) ").strip().lower() != "y":
        return
    if a.task in ("pick_place", "stack2"):
        pp.run(robot, cube, goal)                    # set down: goal.z is the surface under it
    elif a.task == "bowl_drop":
        drop(robot, cube, goal, goal.z + DROP_ABOVE)
    else:
        push_path(robot, cam, cube, path, buf, a.cube)
    print("Done. Measure where the cube ended vs. where it should be.")


if __name__ == "__main__":
    main()
