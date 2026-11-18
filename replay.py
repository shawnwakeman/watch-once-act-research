#!/usr/bin/env python3
"""Replay a demo on the arm: the object-centric trajectory from the pipeline, applied to
wherever the objects are NOW (seen by the robot's Xtion).

DEFAULT = FOLLOW MODE: the robot does only what the demo can't tell it, then follows it.
  The whole motion is planned and checked (follow.py, arm_model.py) BEFORE the arm moves:
  grasp angle + arm posture are chosen so the full demo path stays inside the joint
  limits, and the timing is the demo's, slowed only where the arm's speed / acceleration
  limits require. --check prints that plan and moves nothing.
  pick_place, stack2, bowl_drop
    robot adds:  reach above the cube (in the checked posture), descend, close
    from demo:   everything while the cube moves - path, heights, turns, tilt, timing
    robot adds:  open at the end of the demo path (bowl_drop: at the release lift.py
                 found), lift 10 cm, go back to the start pose
    stack2 / bowl_drop: the path is bent gradually so it ends at the demo's offset to
    the clicked target (printed; ~0 with the demo's layout). Never below the table.
  slide_path
    robot adds:  close the gripper, get behind the cube, fingers PUSH_LOW_M above table
    from demo:   the path, one push per leg (split at corners sharper than 45 deg; an L =
                 2 pushes), the pusher square behind the cube. Before each leg after the
                 first, the camera finds the cube and the leg is re-placed to start
                 there, still ending where the demo's leg ends (offset printed +
                 logged; --no-reaim turns it off).
  If no posture can follow the demo's tilt, only its turn about vertical is followed
  (mode follow-yaw in trials.csv); --yaw-only does that from the start.
  Every change to the demo (smoothing, time stretch, bend, floor, tilt) is printed and
  written to follow_log.jsonl.

--staged = the old mode: scripted pick / lift 10 cm / carry / set down, the demo only
gives the goal (pick_place: displacement; stack2 / bowl_drop: offset to the target;
slide_path: corners, with re-aiming after each push).

The demo comes from lift.py (Mac): copy each run's perception/xtion/trajectory.yaml and
trajectory.npz to the lab PC, e.g. shawn_scripts/demos_l0/pick_place/run01/.

Finding the objects live (held constant across all video-quality levels, so it doesn't
bias the study):
  cube    pink-color segmentation + depth (automatic), or --cube click
  target  automatic, from the 3D points: everything sticking up from the table is split
          into objects; grey block = cube-sized non-pink object, box = biggest one (>= 10
          cm). Target point measured like lift.py did in the demo. --target click to click.
  Debug images + a data snapshot of every run: shawn_scripts/debug/

Usage (inside the container; driver, camera, publish_calibration.py and RViz running):
  python3 replay.py pick_place demos_l0/pick_place/run01 --check        # plan only, moves nothing
  python3 replay.py pick_place demos_l0/pick_place/run01 --hover-only   # check positions by eye
  python3 replay.py pick_place demos_l0/pick_place/run01
  python3 replay.py stack2     demos_l0/stack2/run03
  python3 replay.py bowl_drop  demos_l0/bowl_drop/run02
  python3 replay.py slide_path demos_l0/slide_path/run01
  python3 replay.py pick_place demos_l0/pick_place/run01 --staged      # old mode
After each real run it finds the cube again, asks success y/n + a note, and appends a row
to trials.csv (task, demo, level, mode, error to where the demo ends in cm). --no-log skips that.
Settings (env vars, plus all of pick_place.py's: CUBE_M, SPEED, YAW_DEG, ...):
  PINK_HSV      "140,70,70,179,255,255"  HSV low/high for the cube (OpenCV: H 0-179; low H > high H
                wraps around red). Pick it with pink_check.py
  DROP_ABOVE_M  0.06   bowl_drop --staged: cube bottom this far above the box's clicked point
  PUSH_LOW_M    0.012  slide: fingertips this far above the table while pushing
  PUSH_YAW_OFF  90     slide: gripper twist relative to the push direction, degrees
  GRASP_LOWER_M 0.0    the measured cube top is this much too high: grasp / push this much lower
  GRASP_DX_M, GRASP_DY_M  0   shift the measured cube center in the WORLD (base_link x / y, m):
                       for a camera-side bias
  TOOL_DX_M, TOOL_DY_M    0   the fingers' center vs tool_frame in the GRIPPER's x / y (m):
                       turns with the grasp angle (follow mode)
  GRASP_YAWS    all    follow mode: only try these grasp angles, e.g. "180" or "180,0"
  XY_SCALE      1.05   follow mode: stretch the cube's x-y travel by this factor (empirical
                       correction for placements landing short; same value for every trial)
  XY_SCALE_X, XY_SCALE_Y  = XY_SCALE   the same, separately along the GRIPPER's x / y axes at
                       the grasp (at a 180 deg grasp: gripper x = world -x, gripper y = world +y)
  CUBE_YAW      1      follow mode: line the grasp up with the cube's measured turn (0 = don't)
  CLOSE         0.8    gripper close amount (pick_place.py); 1.0 = squeeze hardest
  CUBE_DETECT   3d     pink cube: color picks the object, 3D points give its shape (as for the
                       grey block); "2d" = the old largest-pink-blob way
  HEIGHT_SRC    cloud  3D points from the point cloud RViz shows (= what you click and what the
                       calibration was fitted to); "depth" = depth image + camera K (the old way)
  SPEED         0.5    follow mode: fraction of the joint speed limits (up to 1.0)
  ACCEL         0.4    follow mode: fraction of the joint acceleration limits
  KNOT_S 0.25, SMOOTH_POS_S 0.10, SMOOTH_ROT_S 0.25   follow path: time between trajectory
                       points / smoothing (see follow.py)
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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint
from tf.transformations import quaternion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick_place as pp                                       # noqa: E402  (motion + gripper code)
import arm_model as am                                        # noqa: E402  (numpy FK / IK of the arm)
import follow as fl                                           # noqa: E402  (demo-following planner)

PINK = [int(v) for v in os.environ.get("PINK_HSV", "140,70,70,179,255,255").split(",")]
DROP_ABOVE = float(os.environ.get("DROP_ABOVE_M", "0.06"))
PUSH_LOW = float(os.environ.get("PUSH_LOW_M", "0.012"))
PUSH_YAW_OFF = float(os.environ.get("PUSH_YAW_OFF", "90"))
GRASP_LOWER = float(os.environ.get("GRASP_LOWER_M", "0.0"))
FOLLOW_CUBE_YAW = os.environ.get("CUBE_YAW", "1") == "1"   # 0 = always grasp at YAW_DEG +- 90s
TOOL_DXY = (float(os.environ.get("TOOL_DX_M", "0")), float(os.environ.get("TOOL_DY_M", "0")))
GRASP_XY = np.array([float(os.environ.get("GRASP_DX_M", "0")), float(os.environ.get("GRASP_DY_M", "0"))])
# follow mode: fractions of the arm's joint speed / acceleration limits (as MoveIt has them).
# SPEED here is NOT capped at 0.5 like pick_place's: a hand moves faster than half this arm.
FOLLOW_SPEED = min(1.0, float(os.environ.get("SPEED", "0.5")))
# 0.4: the arm's firmware rejected 0.7 until ~1.5x slower (= ~0.3 acceleration), every run
ACCEL = min(1.0, float(os.environ.get("ACCEL", "0.4")))
CLOUD_T = os.environ.get("CLOUD_TOPIC", "/camera/depth_registered/points")
CUBE_DETECT = os.environ.get("CUBE_DETECT", "3d")    # 3d = like the grey block; 2d = old color blob
HEIGHT_SRC = os.environ.get("HEIGHT_SRC", "cloud")   # cloud = same 3D points RViz clicks use; depth = depth image + K
# Debug images / data of the last run go into shawn_scripts/debug/ (overwritten each run):
#   scene.png     camera image with every object above the table outlined + what was picked
#   cube_mask.png the pink mask;  objects.png the height mask (what sticks up from the table)
#   snapshot.npz  the camera image + 3D points in base_link - send it to debug detection offline
DEBUG = Path(__file__).resolve().parent / "debug"
RGB_T, DEPTH_T, INFO_T = "/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw", "/camera/rgb/camera_info"


def pink_mask(bgr, rng=None):
    """Pixels inside the HSV range. Hue wraps: low H > high H (e.g. 165..10) means both ends."""
    lo, hi = (rng or PINK)[:3], (rng or PINK)[3:]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if lo[0] <= hi[0]:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    else:
        m = cv2.inRange(hsv, np.array(lo), np.array([179, hi[1], hi[2]])) | \
            cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), np.array(hi))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


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
        self.T = self.pose_of(rgb.header.frame_id)
        # The organized point cloud: the SAME 3D points RViz shows and you click on (and
        # that the touch calibration was fitted to). Using it keeps replay's heights equal
        # to the clicks'. HEIGHT_SRC=depth uses the depth image + K instead.
        self.cloud = None
        if HEIGHT_SRC == "cloud":
            try:
                c = rospy.wait_for_message(CLOUD_T, PointCloud2, timeout=3)
                xyz = cloud_xyz(c)
                if xyz.shape[:2] == self.depth.shape:
                    self.cloud, self.T_cloud = xyz, self.pose_of(c.header.frame_id)
                else:
                    print(f"  (point cloud is {xyz.shape[1]}x{xyz.shape[0]}, image {self.depth.shape[1]}x"
                          f"{self.depth.shape[0]} - using the depth image)")
            except Exception as e:
                print(f"  (no point cloud on {CLOUD_T}: {e} - using the depth image)")
        return self

    def pose_of(self, frame):
        tr = self.buf.lookup_transform(pp.BASE, frame, rospy.Time(0), rospy.Duration(2))
        t, q = tr.transform.translation, tr.transform.rotation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def to_base(self, us, vs, src=None):
        """Pixels -> base_link points (nan where no depth). From the point cloud when it
        is there (src='cloud'), else from the depth image + intrinsics (src='depth')."""
        src = src or ("cloud" if self.cloud is not None else "depth")
        if src == "cloud":
            pc = self.cloud[vs, us]
            return (self.T_cloud[:3, :3] @ pc.T).T + self.T_cloud[:3, 3]
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
        """Top-face center of the pink cube in base_link. CUBE_DETECT=3d (default): the
        same 3D method as the grey block - color only says WHICH object is the cube; its
        shape comes from the 3D points (above the table, grouped from above). 2d: the old
        way (largest pink blob in the image, its depth)."""
        if CUBE_DETECT == "3d":
            c = self.find_pink_cube_3d()
            if c is not None:
                return c
            print("  (3D cube detection found nothing - trying the old 2D color blob)")
        m = pink_mask(self.bgr)
        cv2.imwrite(dbg_path("cube_mask.png"), np.hstack([self.bgr, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)]))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            print(f"  (no pixels in PINK_HSV {PINK} - run pink_check.py to pick the range)")
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = cv2.erode((lab == k).astype(np.uint8), np.ones((5, 5), np.uint8))
        vs, us = np.nonzero(blob)
        pts = self.to_base(us, vs)
        pts = pts[~np.isnan(pts[:, 2])]
        if len(pts) < 30:
            print(f"  (pink blob found, {stats[k, cv2.CC_STAT_AREA]} px, but only {len(pts)} pixels have depth)")
            return None
        top = np.percentile(pts[:, 2], 90)
        sel = pts[pts[:, 2] > top - 0.008]
        xy = np.array([np.median(sel[:, 0]), np.median(sel[:, 1])])
        if GRASP_XY.any():                                   # GRASP_DX_M / GRASP_DY_M fine-tune (default 0)
            print(f"  GRASP_DX_M/DY_M: cube center shifted by ({100 * GRASP_XY[0]:+.1f}, {100 * GRASP_XY[1]:+.1f}) cm")
        xy = xy + GRASP_XY
        cv2.imwrite(dbg_path("cube_mask.png"), np.hstack([self.bgr, cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2BGR)]))
        if self.cloud is not None:                       # how far the two height sources disagree
            pd = self.to_base(us, vs, "depth")
            pd = pd[~np.isnan(pd[:, 2])]
            if len(pd) >= 30:
                print(f"  cube top: point cloud {top:.3f}, depth image {np.percentile(pd[:, 2], 90):.3f} "
                      f"(using the point cloud)")
        c = P(xy[0], xy[1], top)
        c.yaw = face_yaw(sel)
        if c.yaw is not None:
            print(f"  cube turned {c.yaw:.0f} deg (its faces vs the base x axis, 0-90)")
        return c


    def find_pink_cube_3d(self):
        """1. table height under the workspace (median of points near TABLE_Z);
        2. pink pixels (mask widened 2 px so the cube's darker side faces count) whose 3D
           point is 1.2-10 cm above that table and within reach - drops table pixels the
           color mask spills onto and depth mixed at the cube's edges;
        3. group those points on a top-down grid (as for the grey block), keep the biggest
           group = the cube; 4. top-face center + turn, exactly as for the grey block."""
        h, w = self.depth.shape
        vs, us = np.mgrid[0:h, 0:w]
        P3 = self.to_base(us.ravel(), vs.ravel()).reshape(h, w, 3)
        z = np.nan_to_num(P3[:, :, 2], nan=-9.0)
        near = np.abs(z - pp.TABLE_Z) < 0.03
        if near.sum() < 500:
            return None
        table = float(np.median(z[near]))
        x, y = np.nan_to_num(P3[:, :, 0], nan=99.0), np.nan_to_num(P3[:, :, 1], nan=99.0)
        r = np.hypot(x, y)
        pink = cv2.dilate(pink_mask(self.bgr), np.ones((5, 5), np.uint8)) > 0
        m = pink & (z - table > 0.012) & (z - table < 0.10) & (r > 0.12) & (r < 0.80)
        lab = group_from_above(P3, m)
        if lab.max() == 0:
            return None
        sizes = np.bincount(lab.ravel())[1:]
        k = 1 + int(np.argmax(sizes))
        cube = lab == k
        cv2.imwrite(dbg_path("cube_mask.png"), np.hstack([self.bgr, cv2.cvtColor(
            (cube * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
        pts = P3[cube]
        pts = pts[~np.isnan(pts[:, 2])]
        if len(pts) < 30:
            return None
        top = np.percentile(pts[:, 2], 90)
        sel = pts[pts[:, 2] > top - 0.008]
        xy = np.array([np.median(sel[:, 0]), np.median(sel[:, 1])]) + GRASP_XY
        if GRASP_XY.any():
            print(f"  GRASP_DX_M/DY_M: cube center shifted by ({100 * GRASP_XY[0]:+.1f}, {100 * GRASP_XY[1]:+.1f}) cm")
        c = P(xy[0], xy[1], top)
        c.yaw = face_yaw(sel)
        print(f"  cube (3D): {len(pts)} points, top {100 * (top - table):.1f} cm above the table"
              + (f", turned {c.yaw:.0f} deg" if c.yaw is not None else ""))
        return c

    # -------------------------------------------- objects on the table (auto targets) ---
    def scene(self, table_z, cell=0.005):
        """Everything (but the pink cube) that sticks up from the table, as separate objects.
        Works from the 3D points: points 1.2-30 cm above the table and 0.15-0.80 m from the base,
        dropped onto a top-down grid (5 mm cells) and split into connected groups THERE -
        so a block standing in front of the box in the image is still its own object.
        For each: footprint (smallest rectangle around it), top height, saturation, pink."""
        h, w = self.depth.shape
        vs, us = np.mgrid[0:h, 0:w]
        P3 = self.to_base(us.ravel(), vs.ravel()).reshape(h, w, 3)
        self.P3, self.table_z = P3, table_z
        ht = np.nan_to_num(P3[:, :, 2], nan=-9.0) - table_z
        x, y = np.nan_to_num(P3[:, :, 0], nan=99.0), np.nan_to_num(P3[:, :, 1], nan=99.0)
        r = np.hypot(x, y)
        above = (ht > 0.012) & (ht < 0.30) & (r > 0.15) & (r < 0.80)
        # the pink cube is found separately; leave it out so a target right next to it
        # isn't merged with it
        pink = pink_mask(self.bgr) > 0
        above &= cv2.dilate(pink.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
        # top-down grid of the points above the table
        G = int(1.6 / cell)
        gi = np.clip(((x + 0.8) / cell).astype(int), 0, G - 1)
        gj = np.clip(((y + 0.8) / cell).astype(int), 0, G - 1)
        occ = np.zeros((G, G), np.int32)
        np.add.at(occ, (gi[above], gj[above]), 1)
        grid = (occ >= 2).astype(np.uint8)                       # >= 2 points: not a stray pixel
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, glab = cv2.connectedComponents(grid)
        lab = np.where(above, glab[gi, gj], 0)                   # each pixel -> its object
        cv2.imwrite(dbg_path("objects.png"), np.hstack([self.bgr, cv2.cvtColor(
            (above * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
        sat = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
        objs = []
        for k in range(1, n):
            m = lab == k
            if m.sum() < 80:
                continue
            cells = np.argwhere((glab == k) & (occ >= 2))          # footprint from occupied cells
            xy = (cells + 0.5) * cell - 0.8
            (cx, cy), (a_, b_), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
            a_, b_ = a_ + cell * 1000, b_ + cell * 1000            # cell centers -> cell edges
            objs.append(dict(mask=m, area=int(m.sum()), center=np.array([cx, cy]) / 1000,
                             rect=((cx, cy), (a_, b_), ang), long=max(a_, b_) / 1000, short=min(a_, b_) / 1000,
                             top=float(np.percentile(ht[m], 95)), sat=float(np.median(sat[m])),
                             pink=float(pink[m].mean())))
        return objs

    def demo_style_center(self, o):
        """The target point the way lift.py measured it in the demo: median, in the camera
        frame, of ALL the object's visible pixels (SAM's mask there covered the whole object,
        the inside of the box included) -> base_link. Here: every pixel whose x-y falls inside
        the object's footprint (+5 mm) and isn't above its top."""
        (cx, cy), (a_, b_), ang = o["rect"]
        th = math.radians(ang)
        e1, e2 = np.array([math.cos(th), math.sin(th)]), np.array([-math.sin(th), math.cos(th)])
        d = np.nan_to_num(self.P3[:, :, :2]) - np.array([cx, cy]) / 1000
        inside = ((np.abs(d @ e1) <= a_ / 2000 + 0.005) & (np.abs(d @ e2) <= b_ / 2000 + 0.005)
                  & (np.nan_to_num(self.P3[:, :, 2], nan=-9) - self.table_z > -0.01)
                  & (np.nan_to_num(self.P3[:, :, 2], nan=9) - self.table_z < o["top"] + 0.01))
        vs, us = np.nonzero(inside)
        pc, T = self.cam_points(us, vs)
        pc = pc[~np.isnan(pc[:, 2])]
        c = T[:3, :3] @ np.median(pc, 0) + T[:3, 3]
        return P(c[0], c[1], self.table_z + o["top"]), inside

    def find_target(self, task, table_z):
        """stack2: the grey block = a non-pink, low-saturation object about cube-sized
        (footprint 2.5-8 cm, top 2.5-8 cm). bowl_drop: the box = the biggest non-pink
        object with a footprint >= 10 cm. Returns (P or None, why)."""
        objs = self.scene(table_z)
        if task == "stack2":
            cands = [o for o in objs if o["pink"] < 0.2 and 0.025 <= o["long"] <= 0.08 and 0.025 <= o["top"] <= 0.08]
            cands.sort(key=lambda o: abs(o["long"] - pp.CUBE) + abs(o["top"] - pp.CUBE) + o["sat"] / 2550)
            kind = "grey block"
        else:
            cands = sorted([o for o in objs if o["pink"] < 0.2 and o["long"] >= 0.10 and 0.03 <= o["top"] <= 0.30],
                           key=lambda o: -o["area"])
            kind = "box"
        pick = cands[0] if cands else None
        if pick is None:
            t, inside = None, None
        elif task == "stack2":
            t, inside = self.block_top_center(pick)
        else:
            t, inside = self.demo_style_center(pick)
        self.draw_scene(objs, pick, inside, t, kind)
        if pick is None:
            return None, (f"no {kind}-like object above the table (objects seen: "
                          + ", ".join(f"{100 * o['long']:.0f}x{100 * o['short']:.0f} cm, {100 * o['top']:.1f} cm tall"
                                      for o in objs) + ")")
        geo_c = pick["center"]
        self.target_geo = geo_c                               # footprint center, for the stack2 check
        print(f"  {kind}: footprint {100 * pick['long']:.1f} x {100 * pick['short']:.1f} cm, top "
              f"{100 * pick['top']:.1f} cm above the table, footprint center ({geo_c[0]:.3f}, {geo_c[1]:.3f}); "
              f"target point ({t.x:.3f}, {t.y:.3f})"
              + (" = top-face center, measured like the pink cube's" if task == "stack2" else " (as in the demo)"))
        return t, ""

    def block_top_center(self, o):
        """stack2: the block's top-face center, with EXACTLY the pink cube's method (points
        within 8 mm of the top, median x-y). Stacking means cube center over block center;
        measuring both the same way makes the camera's small viewing bias the same for both,
        so it cancels. (The demo-style point - the middle of everything visible - sits
        toward the camera, in front of the block, by ~1-2 cm.)"""
        m = o["mask"]
        pts = self.P3[m]
        ok = ~np.isnan(pts[:, 2])
        top = np.percentile(pts[ok, 2], 90)
        band = m & (np.nan_to_num(self.P3[:, :, 2], nan=-9) > top - 0.008)
        sel = self.P3[band]
        return P(np.median(sel[:, 0]), np.median(sel[:, 1]), top), band

    def project(self, x, y, z):
        inv = np.linalg.inv(self.T)
        pc = inv[:3, :3] @ np.array([x, y, z]) + inv[:3, 3]
        return int(round(self.fx * pc[0] / pc[2] + self.cx)), int(round(self.fy * pc[1] / pc[2] + self.cy))

    def draw_scene(self, objs, pick, inside, t, kind):
        """debug/scene.png: every object above the table outlined in grey with its size, the
        one picked in orange, the pixels used for the target point tinted, the point marked
        (drawn at the object's top height)."""
        img = self.bgr.copy()
        if inside is not None:
            img[inside] = (0.6 * img[inside] + [0, 60, 100]).astype(np.uint8)
        for o in objs:
            cs, _ = cv2.findContours(o["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = (0, 165, 255) if o is pick else (180, 180, 180)
            cv2.drawContours(img, cs, -1, col, 2)
            x, y, _, _ = cv2.boundingRect(cs[0])
            cv2.putText(img, f"{100 * o['long']:.0f}x{100 * o['short']:.0f} h{100 * o['top']:.1f} s{o['sat']:.0f}"
                        + (" pink" if o["pink"] > 0.2 else ""), (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if t is not None:
            cv2.drawMarker(img, self.project(t.x, t.y, t.z), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(img, f"{kind} target", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(dbg_path("scene.png"), img)

    def cam_points(self, us, vs):
        """Pixels -> (points in the camera's optical frame, that frame's pose in base_link)."""
        if self.cloud is not None:
            return self.cloud[vs, us], self.T_cloud
        Z = self.depth[vs, us]
        Z = np.where(Z > 0, Z, np.nan)
        return np.stack([(us - self.cx) * Z / self.fx, (vs - self.cy) * Z / self.fy, Z], 1), self.T

    def save_snapshot(self, cube=None):
        """debug/snapshot.npz: image + base_link 3D points (float16) for offline debugging."""
        h, w = self.depth.shape
        vs, us = np.mgrid[0:h, 0:w]
        P3 = self.to_base(us.ravel(), vs.ravel()).reshape(h, w, 3)
        np.savez_compressed(dbg_path("snapshot.npz"), bgr=self.bgr, xyz=P3.astype(np.float16),
                            T=self.T, K=np.array([self.fx, self.fy, self.cx, self.cy]),
                            cube=np.array([cube.x, cube.y, cube.z]) if cube else np.zeros(0))


def face_yaw(top_pts):
    """Which way the cube's faces point (deg, 0-90), from the outline of its top face.
    None if the outline isn't a clear square."""
    if len(top_pts) < 30:
        return None
    xy = top_pts[:, :2]
    d = np.linalg.norm(xy - np.median(xy, 0), axis=1)
    xy = xy[d <= np.percentile(d, 95)]
    (_, _), (w, h), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
    if min(w, h) < 0.5 * pp.CUBE * 1000:
        return None
    return float(ang % 90)


def group_from_above(P3, mask, cell=0.005):
    """Split the masked pixels into objects by where their 3D points are seen FROM ABOVE
    (5 mm grid cells, >= 2 points per cell, neighbours joined) - so two objects that touch
    in the camera image but stand apart on the table stay separate. Returns a label image
    (0 = none)."""
    G = int(1.6 / cell)
    x, y = np.nan_to_num(P3[:, :, 0], nan=99.0), np.nan_to_num(P3[:, :, 1], nan=99.0)
    gi = np.clip(((x + 0.8) / cell).astype(int), 0, G - 1)
    gj = np.clip(((y + 0.8) / cell).astype(int), 0, G - 1)
    occ = np.zeros((G, G), np.int32)
    np.add.at(occ, (gi[mask], gj[mask]), 1)
    grid = cv2.morphologyEx((occ >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, glab = cv2.connectedComponents(grid)
    return np.where(mask, glab[gi, gj], 0)


def dbg_path(name):
    DEBUG.mkdir(exist_ok=True)
    return str(DEBUG / name)


def cloud_xyz(msg):
    """sensor_msgs/PointCloud2 (organized) -> (H, W, 3) float array, nan where no depth."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": [">f4" if msg.is_bigendian else "<f4"] * 3,
                   "offsets": [off["x"], off["y"], off["z"]], "itemsize": msg.point_step})
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
    a = np.frombuffer(rows.tobytes(), dt).reshape(msg.height, msg.width)
    xyz = np.stack([a["x"], a["y"], a["z"]], -1).astype(float)
    xyz[~np.isfinite(xyz).all(-1)] = np.nan
    return xyz


def locate_target(cam, task, how, buf, cube):
    """Grey block (stack2) / box (bowl_drop), found automatically from the 3D points
    (see Camera.find_target) - or clicked with --target click. Click fallback if not found."""
    if how == "auto":
        t, why = cam.snap().find_target(task, cube.z - pp.CUBE)
        if t is not None:
            print(f"  target: {t}   (debug/scene.png)")
            return t
        print(f"  !! {why} - see debug/scene.png. Click it instead:")
    if task == "stack2":
        p = pp.get_click(buf, "the CENTER OF THE GREY BLOCK'S TOP")
        return P(p.x, p.y, p.z)
    a = pp.get_click(buf, "one TOP CORNER of the box rim")
    b = pp.get_click(buf, "the OPPOSITE top corner of the rim (diagonal)")
    c = P((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
    print(f"  box center (rim midpoint): {c}, rim diagonal {100 * math.hypot(a.x - b.x, a.y - b.y):.1f} cm")
    return c


def table_check(cam, cube):
    """Height of the bare table around the cube, as the camera sees it. If the camera
    sees the cube ~CUBE above the table but the table itself is not at TABLE_Z, the
    calibration is off in z by (seen table - TABLE_Z)."""
    if getattr(cam, "depth", None) is None:
        return
    zs = []
    for ang in np.radians(np.arange(0, 360, 30)):
        for r in (0.06, 0.08):
            z = cam.surface_z_at(cube.x + r * math.cos(ang), cube.y + r * math.sin(ang), cube.z - pp.CUBE)
            if z is not None and z < cube.z - 0.02:          # bare table, not the cube / another object
                zs.append(z)
    if len(zs) < 4:
        return
    t = float(np.median(zs))
    print(f"  table around the cube: z {t:.3f} (TABLE_Z {pp.TABLE_Z}) -> cube {100 * (cube.z - t):.1f} cm above it; "
          f"camera z offset {1000 * (t - pp.TABLE_Z):+.0f} mm"
          + (f"  -> fix: publish_calibration.py --tune, 'z {1000 * (pp.TABLE_Z - t):+.0f}'"
             if abs(t - pp.TABLE_Z) > 0.006 else "  (ok)"))


def locate_cube(cam, how, buf):
    if how == "color":
        c = cam.snap().find_pink_cube()
        if c is None:
            print("  !! no pink cube found (tune PINK_HSV, see debug/cube_mask.png) - click it instead")
        else:
            print(f"  cube top (color+depth): {c}   (debug/cube_mask.png)")
            return c
    p = pp.get_click(buf, "the CENTER OF THE CUBE'S TOP FACE")
    return P(p.x, p.y, p.z)


# --------------------------------------------------------------------- the demo ---
def load_demo(path):
    d = Path(path)
    summ = yaml.safe_load((d / "trajectory.yaml").read_text())
    npz = d / "trajectory.npz"
    data = dict(np.load(npz)) if npz.exists() else None
    return summ, data


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


# ------------------------------------------------- follow mode (the default) ---
# All planning is in follow.py + arm_model.py (plain numpy, also run offline by
# test_follow.py). This part only talks to the robot.
def pose_to_mat(p):
    T = quaternion_matrix([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return T


def geo():
    return dict(cube=pp.CUBE, tip=pp.TIP, approach=pp.APPROACH, clear=pp.CLEAR,
                grip_depth=pp.GRIP_DEPTH, push_low=PUSH_LOW, push_yaw_off=PUSH_YAW_OFF,
                tool_dx=TOOL_DXY[0], tool_dy=TOOL_DXY[1], end_floor=END_FLOOR[0])


END_FLOOR = [None]       # stack2: the cube must end with its bottom at least this high (block top + 4 mm)


def live_limits():
    """Joint speed / acceleration limits MoveIt uses for this arm (falls back to the
    values built into arm_model.py)."""
    v, a = am.VMAX.copy(), am.AMAX.copy()
    try:
        lim = rospy.get_param(rospy.get_namespace() + "robot_description_planning/joint_limits")
        for i in range(6):
            j = lim.get(f"joint_{i + 1}", {})
            v[i] = float(j.get("max_velocity", v[i]))
            a[i] = float(j.get("max_acceleration", a[i]))
    except Exception:
        print("  (MoveIt joint limits not found - using arm_model.py's)")
    return dict(vmax=v, amax=a)


def joints_now(robot):
    return np.array(robot.arm.get_current_joint_values(), float)


def model_matches(robot):
    """Safety: our arm model must agree with MoveIt about where the tool is right now."""
    T = pose_to_mat(robot.arm.get_current_pose().pose)
    M = am.fk(joints_now(robot))
    d_mm = 1000 * np.linalg.norm(T[:3, 3] - M[:3, 3])
    d_deg = fl.rot_deg(T[:3, :3].T @ M[:3, :3])
    if d_mm > 3 or d_deg > 1:
        print(f"  !! arm model disagrees with MoveIt by {d_mm:.1f} mm / {d_deg:.1f} deg - follow mode refused "
              "(is the end effector link tool_frame?). Use --staged.")
        return False
    return True


class Arm:
    """Sends a joint trajectory straight to the driver's FollowJointTrajectory action
    (not MoveIt's execute(), which aborts with TIMED_OUT when the arm runs behind).

    Two things about the Kinova driver this handles:
      - The arm checks the whole trajectory before moving and rejects it if it is too
        fast for its own (firmware) limits. Then nothing moved: retry 15% slower, and keep
        that slow-down for the rest of the run so it isn't rejected again every time.
      - Its default end tolerance is 0.005 rad (0.3 deg) per joint; landing a hair outside
        it is reported as a failure. We allow 0.02 rad (1.1 deg) and, whatever the
        status, judge the move by where the arm actually ended up - and if it stopped more
        than 0.17 deg from the last point (it tends to stop short on long moves), finish
        the move with one short, slow move to the exact end."""
    END_TOL = 0.02
    SETTLE_TOL = 0.003                        # rad (0.17 deg): further than this from the end -> finish the move

    def __init__(self, robot):
        self.robot = robot
        self.slow = 1.0                       # learned firmware slow-down, kept for this run
        self.last_end_mm = None
        topics = [t for t, _ in rospy.get_published_topics() if t.endswith("follow_joint_trajectory/status")
                  and "gripper" not in t]
        self.client = None
        if topics:
            name = topics[0][:-len("/status")]
            self.client = actionlib.SimpleActionClient(name, FollowJointTrajectoryAction)
            if not self.client.wait_for_server(rospy.Duration(5)):
                self.client = None
        if self.client is None:
            print("  (no joint trajectory action found - follow mode can plan but not move)")

    def run(self, q, t, label, settle=True):
        """Run trajectory points q at times t (the first point = where the arm is now).
        Returns 'ok', 'rejected' (nothing moved) or 'failed'."""
        if self.client is None:
            return "failed"
        q, t0_plan = np.array(q, float), np.asarray(t, float)
        for attempt in range(6):
            t = t0_plan * self.slow
            q[0] = joints_now(self.robot)         # the driver wants the first point = where the arm is
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = self.robot.arm.get_active_joints()
            vel = np.gradient(q, t, axis=0)
            vel[0] = vel[-1] = 0.0
            for qi, vi, ti in zip(q, vel, t):
                pt = JointTrajectoryPoint()
                pt.positions, pt.velocities = [float(x) for x in qi], [float(x) for x in vi]
                pt.time_from_start = rospy.Duration(float(ti))
                goal.trajectory.points.append(pt)
            for n in goal.trajectory.joint_names:
                tol = JointTolerance()
                tol.name, tol.position = n, self.END_TOL
                goal.goal_tolerance.append(tol)
            dur = float(t[-1])
            goal.goal_time_tolerance = rospy.Duration(max(3.0, 0.5 * dur))
            before, t0 = joints_now(self.robot), rospy.get_time()
            self.client.send_goal(goal)
            done = self.client.wait_for_result(rospy.Duration(dur + 2.0 + max(5.0, 0.5 * dur)))
            took = rospy.get_time() - t0
            if not done:
                self.client.cancel_goal()
                print(f"  {label}: no result after {took:.1f} s - cancelled")
                return "failed"
            state = self.client.get_state()
            rospy.sleep(0.2)
            now = joints_now(self.robot)
            off = float(np.abs(now - q[-1]).max())
            slow_txt = f", {self.slow:.2f}x slower than planned" if self.slow > 1.001 else ""
            if state == 3 or off < self.END_TOL:
                end_mm = 1000 * np.linalg.norm(am.fk(now)[:3, 3] - am.fk(q[-1])[:3, 3])
                print(f"  {label}: done in {took:.1f} s (planned {dur:.1f} s{slow_txt} + ~1.5 s driver start-up); "
                      f"stopped {end_mm:.1f} mm / {math.degrees(off):.2f} deg from the planned end")
                if settle:
                    self.last_end_mm = end_mm         # how short the trajectory itself stopped
                if off > self.SETTLE_TOL and settle:
                    # The arm finishes its trajectory a little short of the last point (more
                    # on long moves). Finish the move: one short, slow move to the exact end.
                    self.run([now, q[-1]], [0.0, 1.0], "  finish (to the exact end)", settle=False)
                return "ok"
            res = self.client.get_result()
            msg = getattr(res, "error_string", "") if res is not None else ""
            if np.abs(now - before).max() > 0.01:
                print(f"  {label}: FAILED part-way after {took:.1f} s, {math.degrees(off):.0f} deg from the end "
                      f"(status {state}{', ' + msg if msg else ''}; see the driver terminal)")
                return "failed"
            self.slow *= 1.15
            print(f"  {label}: the arm rejected the trajectory before moving (status {state}"
                  f"{', ' + msg if msg else ''}) - now {self.slow:.2f}x slower for the rest of this run")
        return "rejected"


def show_plan(p, info=None):
    print(fl.report(p))
    if info is not None:
        print(f"    changes to the demo: smoothing moved the path up to {info['smooth_mm']:.1f} mm / "
              f"{info['smooth_deg']:.1f} deg | bent {100 * info['bend_m']:.1f} cm toward the target | "
              f"{info['floored']} frames raised to the table | "
              + (f"end raised {info['end_raised_mm']:.0f} mm to clear the block | " if info.get("end_raised_mm") else "")
              + ("x-y travel scaled x{:.3f} (gripper x) / x{:.3f} (gripper y) | ".format(*info["xy_scale"])
                 if tuple(info.get("xy_scale", (1.0, 1.0))) != (1.0, 1.0) else "")
              + "rotation: "
              + ("turn about vertical only (tilt dropped)" if info.get("yaw_only") else "full"))


def plan_summary(p, info=None, **more):
    """The numbers that describe how the run differs from the demo (for follow_log.jsonl)."""
    keys = ["label", "n", "demo_dur", "dur", "k", "k_max", "path_m", "err_cm_max", "err_cm_mean", "err_deg_max",
            "v_peak", "a_peak", "margin_deg"]
    d = {k: (round(float(p[k]), 4) if not isinstance(p[k], str) else p[k]) for k in keys if k in p}
    if info is not None:
        d.update({k: round(float(info[k]), 4) for k in ("smooth_mm", "smooth_deg", "bend_m", "floored",
                                                          "end_raised_mm") if k in info})
        if "xy_scale" in info:
            d["xy_scale_gripper_xy"] = [float(v) for v in info["xy_scale"]]
        d["yaw_only"] = bool(info.get("yaw_only"))
    d.update(knot_s=fl.KNOT_S, smooth_pos_s=fl.SMOOTH_POS_S, smooth_rot_s=fl.SMOOTH_ROT_S,
             speed=FOLLOW_SPEED, accel=ACCEL, **more)
    return d


def carry_setup(a, summ, data, grip, tgt):
    """Everything needed to plan a carry: demo span, target end point, cube top now."""
    Tb, times = data["T_base"], data["times"]
    g, e = fl.moving_span(Tb, ends_at_release=(a.task == "bowl_drop"))
    end_xy = None
    if tgt is not None:
        ex, ey, _ = summ["end_minus_target_m"]
        end_xy = (tgt.x + ex, tgt.y + ey)
    return Tb, times, g, e, (grip.x, grip.y, grip.z), end_xy


def plan_carry(robot, a, summ, data, grip, tgt, lim):
    """Choose grasp angle + arm posture so the WHOLE demo path passes every check, from
    where the arm is now - before anything moves. Returns (plan, info, (yaw, yaw_only))."""
    Tb, times, g, e, top, end_xy = carry_setup(a, summ, data, grip, tgt)
    # grasp across two opposite faces: the four grasp angles follow the cube's own turn
    yaw0 = pp.YAW_DEG if getattr(grip, "yaw", None) is None else (grip.yaw if FOLLOW_CUBE_YAW else pp.YAW_DEG)
    best, pick, lines = fl.choose_carry(Tb, times, g, e, top, geo(), yaw0, end_xy, joints_now(robot),
                                        FOLLOW_SPEED, ACCEL, yaw_only=a.yaw_only, **lim)
    print(f"  demo frames {g}-{e} ({times[e] - times[g]:.1f} s) | settings: point every {fl.KNOT_S} s, smoothing "
          f"pos {fl.SMOOTH_POS_S} s / rot {fl.SMOOTH_ROT_S} s, SPEED {FOLLOW_SPEED}, ACCEL {ACCEL}")
    for ln in lines:
        print("    " + ln)
    if best is None:
        return None, None, None
    yaw, yaw_only = pick
    gz, _ = fl.grasp_heights(top[2], geo())
    _, _, info = fl.carry_poses(Tb, times, g, e, top, geo(), fl.down_pose(top[0], top[1], gz, yaw), yaw_only, end_xy)
    info["yaw_only"] = yaw_only
    show_plan(best, info)
    if tgt is not None:
        o = info["cube_path"][-1]
        geo_c = getattr(a, "target_geo", None)
        print(f"  planned cube end: center ({o[0]:.3f}, {o[1]:.3f}), bottom z {o[2] - pp.CUBE / 2:.3f} | target point "
              f"({tgt.x:.3f}, {tgt.y:.3f}), top z {tgt.z:.3f} | demo offset to it "
              f"({100 * (end_xy[0] - tgt.x):+.1f}, {100 * (end_xy[1] - tgt.y):+.1f}) cm"
              + (f" | from the {('block' if a.task == 'stack2' else 'box')}'s footprint center: "
                 f"({100 * (o[0] - geo_c[0]):+.1f}, {100 * (o[1] - geo_c[1]):+.1f}) cm" if geo_c is not None else ""))
    return best, info, pick


def after_failure(robot, what):
    print(f"  FAILED: {what}. The arm is stopped where it is.")
    if input("  open the gripper here? [y/N] ").strip().lower() == "y":
        robot.open_gripper()
    robot.quat = pp.down_with_yaw(pp.YAW_DEG)


def follow_carry(robot, arm, a, summ, data, grip, tgt, lim, best, pick):
    """Reach + grasp (the only part the demo can't give), then ride along the demo path.
    Returns (ok, plan summary)."""
    Tb, times, g, e, top, end_xy = carry_setup(a, summ, data, grip, tgt)
    yaw, yaw_only = pick
    gz, _ = fl.grasp_heights(top[2], geo())
    start = robot.arm.get_current_joint_values()
    robot.quat = pp.down_with_yaw(yaw)
    for name, fn in [("open", robot.open_gripper),
                     ("above the cube, in the checked posture", lambda: robot.go_joints(list(best["q_pre"]))),
                     ("open", robot.open_gripper),
                     ("descend", lambda: robot.straight_to(gz)),
                     ("close", robot.close_gripper)]:
        print(f"  -> {name}")
        if not fn():
            after_failure(robot, f"'{name}'")
            return False, plan_summary(best, status="failed before the demo part: " + name)
    # Plan again from where the gripper REALLY is (a few mm / tenths of a degree from the
    # plan) - same code, same checks.
    q0 = joints_now(robot)
    poses, tt, info = fl.carry_poses(Tb, times, g, e, top, geo(), am.fk(q0), yaw_only, end_xy)
    info["yaw_only"] = yaw_only
    p = fl.plan_path(poses, tt, q0, FOLLOW_SPEED, ACCEL, top[2] - pp.CUBE, **lim)
    p["label"] = best["label"]
    if not p["ok"]:
        after_failure(robot, "the path from the real grasp pose fails the checks: " + "; ".join(p["why"]))
        return False, plan_summary(best, info, status="failed re-check: " + "; ".join(p["why"]))
    show_plan(p, info)
    print("  -> follow the demo")
    r = arm.run(p["q"], p["t"], "demo path")
    stop_mm = arm.last_end_mm
    if r != "ok":
        after_failure(robot, f"following the demo ({r})")
        return False, plan_summary(p, info, status=r)
    final_mm = 1000 * np.linalg.norm(am.fk(joints_now(robot))[:3, 3] - am.fk(p["q"][-1])[:3, 3])
    print(f"  at release: {final_mm:.1f} mm from the planned end (the trajectory itself stopped {stop_mm:.1f} mm short)")
    if a.pause_release:
        input("  holding the cube at the end of the demo path - look where it is over the block "
              "(and whether it's tilted), then Enter to release ")
    print("  -> release")
    robot.open_gripper()
    print("  -> lift + return")
    cur = robot.arm.get_current_pose().pose
    robot.quat = [cur.orientation.x, cur.orientation.y, cur.orientation.z, cur.orientation.w]
    robot.straight_to(cur.position.z + pp.APPROACH)
    robot.quat = pp.down_with_yaw(pp.YAW_DEG)
    robot.go_joints(start)
    return True, plan_summary(p, info, status="ok", stop_short_mm=round(stop_mm or 0, 1),
                              release_off_mm=round(final_mm, 1))


def plan_pushes(robot, data, grip, lim):
    """Plan every push of the slide from the demo path, before anything moves."""
    Tb, times = data["T_base"], data["times"]
    g, e = fl.moving_span(Tb, ends_at_release=False)
    segs = fl.push_segments(Tb, g, e, (grip.x, grip.y), times)
    print(f"  demo frames {g}-{e}: {len(segs)} leg(s) - one push per leg, split at corners sharper than 45 deg")
    plans, q = [], joints_now(robot)
    for k, seg in enumerate(segs, 1):
        opts, meta = fl.push_options(seg, grip.z, geo())
        best, lines = fl.choose(opts, q, FOLLOW_SPEED, ACCEL, grip.z - pp.CUBE, **lim)
        print(f"  push {k}: {100 * np.sum(np.linalg.norm(np.diff(seg[0], axis=0), axis=1)):.1f} cm")
        for ln in lines:
            print("    " + ln)
        if best is None:
            return None, segs
        show_plan(best)
        plans.append((best, opts, meta[best["label"]]))
        q = best["q"][-1]
    return plans, segs


def follow_push(robot, arm, cam, buf, a, grip, lim, plans, segs):
    """Get behind the cube (the only part the demo can't give), then push it along the
    demo's own path.
    One push per leg of the demo path (an L = 2 pushes). The cube drifts a bit during a
    push, so before each leg after the first the camera finds it again and that leg is
    re-placed to start at the cube; the offset fades out along the leg so it still ends
    where the demo's leg ends. The offset is printed and logged.
    --no-reaim pushes the planned path blind."""
    start = robot.arm.get_current_joint_values()
    logs = []
    pos = robot.send_grip(1.0)                    # fully closed: no gap between the fingers to catch a corner
    print(f"  gripper fully closed for pushing ({pos:.2f})")
    for k, (best, opts, meta) in enumerate(plans, 1):
        shift = np.zeros(2)
        if k > 1 and not a.no_reaim:
            seen = None
            try:
                seen = cam.snap().find_pink_cube()
            except Exception as e:
                print(f"  (camera: {e})")
            if seen is None:
                print(f"  push {k}: cube not found by the camera - pushing the planned path")
            else:
                seg2, shift = fl.retarget_leg(segs[k - 1], (seen.x, seen.y))
                print(f"  push {k}: cube is {100 * np.linalg.norm(shift):.1f} cm from where the demo path "
                      f"expects it - this leg now starts at the cube and still ends where the demo's leg ends")
                if np.linalg.norm(shift) > 0.002:
                    opts2, meta2 = fl.push_options(seg2, grip.z, geo())
                    b2, lines = fl.choose(opts2, joints_now(robot), FOLLOW_SPEED, ACCEL, grip.z - pp.CUBE, **lim)
                    if b2 is None:
                        after_failure(robot, f"push {k}: no plan from where the cube is now ({lines[0]})")
                        return False, logs + [plan_summary(best, status="failed: re-aim, no plan")]
                    best, opts, meta = b2, opts2, meta2[b2["label"]]
        yaw, behind, low, safe, last, u = meta
        robot.quat = pp.down_with_yaw(yaw)
        poses, tt = [o for o in opts if o[0] == best["label"]][0][3]()
        for name, fn in [("above the start, in the checked posture", lambda: robot.go_joints(list(best["q_pre"]))),
                         ("lower", lambda: robot.straight_to(low))]:
            print(f"  -> push {k}: {name}")
            if not fn():
                after_failure(robot, f"push {k} '{name}'")
                return False, logs + [plan_summary(best, status="failed: " + name)]
        q0 = joints_now(robot)
        poses = [am.fk(q0)] + list(poses[1:])
        p = fl.plan_path(poses, tt, q0, FOLLOW_SPEED, ACCEL, grip.z - pp.CUBE, **lim)
        p["label"] = best["label"]
        if not p["ok"]:
            after_failure(robot, f"push {k} fails the re-check: " + "; ".join(p["why"]))
            return False, logs + [plan_summary(best, status="failed re-check")]
        r = arm.run(p["q"], p["t"], f"push {k} along the demo path")
        logs.append(plan_summary(p, status=r, push=k, reaim_shift_cm=round(100 * float(np.linalg.norm(shift)), 2)))
        if r != "ok":
            after_failure(robot, f"push {k} ({r})")
            return False, logs
        # back off 2 cm along the push direction (so lifting doesn't drag the cube), then raise
        q1 = joints_now(robot)
        back = am.fk(q1)
        back[:2, 3] -= u * 0.02
        qb = am.ik(back, q1)
        if qb is not None:
            arm.run([q1, qb], [0.0, 1.5], "back off")
        robot.straight_to(safe)
    robot.quat = pp.down_with_yaw(pp.YAW_DEG)
    robot.go_joints(start)
    return True, logs


def write_follow_log(a, entries):
    """One line per follow run with everything that differs from the demo."""
    import json
    import time
    path = Path(__file__).resolve().parent / "follow_log.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": a.task, "demo": a.demo,
                            "level": a.level, "plans": entries}) + "\n")
    print(f"  plan details -> {path}")


# ------------------------------------------------------------------ trial log ---
def log_trial(a, cam, cube0, goal, mode):
    """After the arm is back home: find the cube again, compare with the goal, ask for the
    verdict, append one row to trials.csv (next to this script)."""
    import csv
    import time
    rospy.sleep(1.0)
    end = None
    try:
        end = cam.snap().find_pink_cube()
    except Exception as e:                      # camera hiccup: still log the verdict
        print(f"  (couldn't re-measure the cube: {e})")
    if end is None:
        print("  cube not found by color now (hidden in the box / out of view)")
        exy = ez = None
    else:
        exy = 100 * math.hypot(end.x - goal.x, end.y - goal.y)
        # stack2 / pick_place: goal.z is the surface under the cube -> top should be at goal.z + CUBE
        ez = 100 * (end.z - (goal.z + pp.CUBE)) if a.task in ("pick_place", "stack2") else None
        print(f"  cube now {end}: {exy:.1f} cm from the goal in x-y"
              + (f", top {ez:+.1f} cm vs expected" if ez is not None else ""))
    ans = input("  success? [y/n]: ").strip().lower()
    note = input("  note (Enter = none): ").strip()
    path = Path(__file__).resolve().parent / "trials.csv"
    if path.exists():                                   # older file without a mode column
        rows = list(csv.reader(open(path)))
        if rows and "mode" not in rows[0]:
            rows = [rows[0][:4] + ["mode"] + rows[0][4:]] + [r[:4] + ["staged"] + r[4:] for r in rows[1:]]
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(rows)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "level", "task", "demo", "mode", "success", "err_xy_cm", "err_z_cm",
                        "cube_x", "cube_y", "cube_z", "goal_x", "goal_y", "goal_z",
                        "end_x", "end_y", "end_z", "note"])
        r = lambda v: "" if v is None else round(v, 4)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), a.level, a.task, a.demo, mode,
                    int(ans == "y"), r(exy), r(ez), r(cube0.x), r(cube0.y), r(cube0.z),
                    r(goal.x), r(goal.y), r(goal.z),
                    r(end.x if end else None), r(end.y if end else None), r(end.z if end else None), note])
    print(f"  logged -> {path}")


# ------------------------------------------------------------------------ main ---
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["pick_place", "stack2", "bowl_drop", "slide_path"])
    ap.add_argument("demo", help="folder with the run's trajectory.yaml + trajectory.npz")
    ap.add_argument("--cube", choices=["color", "click"], default="color")
    ap.add_argument("--target", choices=["auto", "click"], default="auto",
                    help="grey block / box: found automatically from the 3D points (auto) or clicked")
    ap.add_argument("--staged", action="store_true",
                    help="old mode: scripted pick / carry / place, the demo only gives the goal")
    ap.add_argument("--yaw-only", action="store_true",
                    help="follow mode: follow the demo's turn about vertical but not its tilt")
    ap.add_argument("--check", action="store_true",
                    help="follow mode: plan everything, print the report, move NOTHING")
    ap.add_argument("--pause-release", action="store_true",
                    help="stop at the end of the demo path and wait for Enter before opening (to inspect)")
    ap.add_argument("--no-reaim", action="store_true",
                    help="slide_path: don't look for the cube again between pushes")
    ap.add_argument("--hover-only", action="store_true", help="just hover above the cube and the end")
    ap.add_argument("--level", default="L0", help="video-quality level of this demo, for trials.csv")
    ap.add_argument("--no-log", action="store_true", help="don't measure / log the trial")
    a = ap.parse_args()

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("replay")
    buf = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buf)  # keep a reference
    rospy.sleep(1.0)
    robot = pp.Robot()
    cam = Camera(buf)
    summ, data = load_demo(a.demo)
    if data is None:
        raise SystemExit("needs trajectory.npz next to trajectory.yaml")
    print(f"Demo {a.demo}: displacement {summ['displacement_m']} m, end rotation {summ['end_rotation_deg']} deg")
    if a.task == "bowl_drop" and not summ.get("trajectory_ends_at_release"):
        print("  !! this bowl_drop demo was made before lift.py cut it at the release - re-run lift.py")

    cube = locate_cube(cam, a.cube, buf)
    print(f"  cube top z {cube.z:.3f} = {100 * (cube.z - pp.TABLE_Z):.1f} cm above TABLE_Z "
          f"(a {pp.CUBE * 100:.0f} cm cube should be ~{pp.CUBE * 100:.0f})")
    table_check(cam, cube)
    try:
        cam.save_snapshot(cube)
        if a.task in ("pick_place", "slide_path"):
            cam.draw_scene(cam.scene(cube.z - pp.CUBE), None, None, None, "")
    except Exception as e:
        print(f"  (debug snapshot failed: {e})")
    grip = P(cube.x, cube.y, cube.z - GRASP_LOWER)   # GRASP_LOWER_M: measured top is this much too high
    grip.yaw = getattr(cube, "yaw", None)
    if GRASP_LOWER:
        print(f"  GRASP_LOWER_M: grasping / pushing {GRASP_LOWER * 100:.1f} cm lower")

    tgt = None
    if a.task in ("stack2", "bowl_drop"):
        if "end_minus_target_m" not in summ:
            raise SystemExit("This demo has no target (run perceive.py with --target, then lift.py).")
        tgt = locate_target(cam, a.task, a.target, buf, cube)
        a.target_geo = getattr(cam, "target_geo", None) if a.target == "auto" else None

    if a.task == "stack2" and tgt is not None:
        END_FLOOR[0] = tgt.z + 0.004
        print(f"  block top z {tgt.z:.3f} -> the cube's bottom will end at least 4 mm above it")

    if a.staged:
        run_staged(a, robot, cam, buf, summ, data, cube, grip, tgt)
        return

    # ---- follow mode: plan and check EVERYTHING first; nothing has moved yet
    if not model_matches(robot):
        return
    lim = live_limits()
    if a.task == "slide_path":
        plans, segs = plan_pushes(robot, data, grip, lim)
        ok_plan = plans is not None
        end = segs[-1][0][-1]
        goal = P(end[0], end[1], cube.z - pp.CUBE)
        pts = [P(p[0], p[1], cube.z) for sg in segs for p in sg[0][::10]]
    else:
        best, info, pick = plan_carry(robot, a, summ, data, grip, tgt, lim)
        ok_plan = best is not None
        if ok_plan:
            o = info["cube_path"][-1]
            goal = P(o[0], o[1], o[2] - pp.CUBE / 2 + GRASP_LOWER)   # surface under the cube's end spot
            pts = [P(*p) for p in info["cube_path"][::10]]
    if not ok_plan:
        print("  NO PLAN passes the checks - nothing moved. Move the cube closer to where it started in "
              "the demo, or use --staged.")
        return
    if a.check:
        print("  --check: plan passes every check. Nothing moved.")
        return
    if not check_reach(cube, *pts):
        return
    if a.hover_only:
        print("  hover check: above the cube, then above where the demo path ends")
        robot.move_to(cube.x, cube.y, cube.z + 0.05 + pp.TIP)
        input("  above the cube - Enter for the end ")
        robot.move_to(goal.x, goal.y, goal.z + pp.CUBE + 0.05 + pp.TIP)
        print("  above the end. Check by eye, then run without --hover-only.")
        return
    if input(f"Follow the demo for {a.task}? [y/N] (gamepad RB = E-stop) ").strip().lower() != "y":
        return
    arm = Arm(robot)
    if a.task == "slide_path":
        ok, entries = follow_push(robot, arm, cam, buf, a, grip, lim, plans, segs)
        mode = "follow"
    else:
        ok, entry = follow_carry(robot, arm, a, summ, data, grip, tgt, lim, best, pick)
        entries, mode = [entry], ("follow-yaw" if pick[1] else "follow")
    print("Done." if ok else "Stopped early.")
    if not a.no_log:
        write_follow_log(a, entries)
        log_trial(a, cam, cube, goal, mode if ok else mode + ":FAILED")


def run_staged(a, robot, cam, buf, summ, data, cube, grip, tgt):
    """--staged: scripted pick / carry / place; the demo only gives the goal."""
    if a.task == "pick_place":
        dx, dy = summ["displacement_m"][:2]
        gx, gy = cube.x + dx, cube.y + dy
        gz = cam.surface_z_at(gx, gy, cube.z - pp.CUBE)
        gz = gz if gz is not None and abs(gz - (cube.z - pp.CUBE)) < 0.03 else cube.z - pp.CUBE
        goal = P(gx, gy, gz)                         # the table surface where it goes
        print(f"  goal (table under the cube's new spot): {goal}  = cube + demo displacement")
    elif a.task in ("stack2", "bowl_drop"):
        ex, ey, _ = summ["end_minus_target_m"]
        goal = P(tgt.x + ex, tgt.y + ey, tgt.z)
        print(f"  target {P(tgt.x, tgt.y, tgt.z)} + demo offset ({ex * 100:.1f}, {ey * 100:.1f}) cm -> goal {goal}")
    else:  # slide_path
        xyz = data["T_base"][:, :3, 3]
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
    if input(f"Run {a.task} (staged)? [y/N] (gamepad RB = E-stop) ").strip().lower() != "y":
        return
    if a.task in ("pick_place", "stack2"):
        pp.run(robot, grip, goal)                    # set down: goal.z is the surface under it
    elif a.task == "bowl_drop":
        drop(robot, grip, goal, goal.z + DROP_ABOVE)
    else:
        push_path(robot, cam, grip, path, buf, a.cube)   # push height from the corrected top too
    print("Done.")
    if not a.no_log:
        log_trial(a, cam, cube, goal, "staged")


if __name__ == "__main__":
    main()