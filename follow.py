"""Demo-following for replay.py - pure numpy (no ROS), so the exact code that plans the
robot's motion also runs offline in test_follow.py.

THE IDEA
  The demo gives the cube's pose over time. Once the robot holds the cube, the gripper
  rides along with it:      E(t) = D(t) @ E_grasp
  D(t) = the demo cube's rigid motion since the grasp frame, moved to where the cube is
  now. So path, heights, turns, tilt and timing come from the demo.

HOW IT IS TURNED INTO AN ARM MOTION (and why)
  1. smooth    The tracked path still wobbles by mm / degrees. Gaussian smoothing over
               hand time scales (SMOOTH_POS_S / SMOOTH_ROT_S), start and end pinned.
  2. knots     Sample the gripper path every KNOT_S of demo time. Kinova's driver throws
               away MoveIt's velocities and gives the arm only (joint angles, duration)
               per point; the arm's firmware makes the motion in between and rejects
               segments needing too much acceleration. Few, evenly timed points on a
               smooth path are what it follows without stop-and-go.
  3. joints    Inverse kinematics knot by knot, each seeded from the one before, so the
               arm stays on one 'branch' (no wrist flips). Done for several grasp angles
               (a cube looks the same turned 90 deg) and arm postures BEFORE touching
               the cube; the first one where the whole path passes every check is used.
  4. timing    Keep the demo's timing; where a joint would exceed its speed /
               acceleration limit, lengthen just those segments until it fits. The
               total stretch k and the biggest local stretch are printed and logged.
  5. checks    joint-limit margin, no jumps, nothing below the table, path error vs the
               smoothed demo. Fail -> the arm never moves.

WHAT THE ROBOT ADDS BEYOND THE DEMO (everything is printed + logged)
  reach + grasp / getting behind the cube to push; release, lift, go home;
  smoothing (1); time stretch k (4); a floor so the cube never goes below the table;
  stack2 / bowl_drop: bending the path so it ends at the demo's offset to the target;
  tilt dropped (yaw only) if no posture can follow the tilt.
"""
import math
import os

import numpy as np

import arm_model as am

MOVE_START_M = 0.005     # the demo cube counts as moving once it's this far from its start
SETTLE_M = 0.003         # ... and as settled once it stays this close to its final position
# Settings (env vars), printed with every plan:
KNOT_S = float(os.environ.get("KNOT_S", "0.25"))              # one trajectory point per 0.25 s of demo
SMOOTH_POS_S = float(os.environ.get("SMOOTH_POS_S", "0.8"))  # position smoothing (gaussian sigma, s)
SMOOTH_ROT_S = float(os.environ.get("SMOOTH_ROT_S", "0.2"))  # rotation smoothing (sigma, s)
# Empirical correction: stretch the cube's x-y travel (from its start), separately along the
# GRIPPER's x and y axes at the grasp (tool_frame; at a 180 deg grasp gripper x = world -x,
# gripper y = world +y). 1.0 = the demo's path as is. XY_SCALE sets both; XY_SCALE_X /
# XY_SCALE_Y override one axis. Measured on the arm: placements land a few % short.
XY_SCALE = float(os.environ.get("XY_SCALE", "1.05"))
XY_SCALE_X = float(os.environ.get("XY_SCALE_X", "1.06"))
XY_SCALE_Y = float(os.environ.get("XY_SCALE_Y", "1.05"))
LIMIT_MARGIN_DEG = 5.0  # every joint stays this far from its limit
MAX_STEP_RAD = 0.6       # a bigger joint change between two knots = a branch jump
MAX_ERR_M, MAX_ERR_DEG = 0.01, 5.0     # allowed path error vs the smoothed demo
BODY_CLEAR_M = 0.02      # wrist / elbow joints stay this far above the table


def moving_span(Tb, ends_at_release):
    """First / last frame of the part of the demo where the cube moves (g, e)."""
    pos = Tb[:, :3, 3]
    mv = np.nonzero(np.linalg.norm(pos - pos[0], axis=1) > MOVE_START_M)[0]
    if not len(mv):
        raise ValueError("the cube never moves in this demo")
    g = max(0, int(mv[0]) - 2)
    if ends_at_release:                       # lift.py already ended it at the release
        return g, len(pos) - 1
    st = np.nonzero(np.linalg.norm(pos - pos[-1], axis=1) > SETTLE_M)[0]
    e = min(len(pos) - 1, int(st[-1]) + 2) if len(st) else len(pos) - 1
    return g, max(e, g + 1)


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_deg(R):
    return math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def gauss(x, sigma):
    """Gaussian smoothing along axis 0 (edges padded with the end values)."""
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def so3_log(R):
    th = math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


def so3_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def smooth_segment(Tb, times):
    """Extra smoothing of the moving part of the demo; start and end poses kept as they were."""
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    pos = Tb[:, :3, 3]
    ps = gauss(pos, SMOOTH_POS_S / dt)
    s = arc_fraction(pos)[:, None]
    ps += (1 - s) * (pos[0] - ps[0]) + s * (pos[-1] - ps[-1])        # pin the ends
    R0 = Tb[0, :3, :3]
    rv = np.array([so3_log(R0.T @ T[:3, :3]) for T in Tb])          # rotation since the start
    rs = gauss(rv, SMOOTH_ROT_S / dt)
    rs += s * (rv[-1] - rs[-1]) + (1 - s) * (rv[0] - rs[0])
    out = Tb.copy()
    for i in range(len(Tb)):
        out[i, :3, :3] = R0 @ so3_exp(rs[i])
        out[i, :3, 3] = ps[i]
    return out


def arc_fraction(pts):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(pts))


def carry_motion(Tb, times, g, e, origin_now, yaw_only=False, end_xy=None, floor_z=None, half_height=0.025,
                 end_floor_z=None, grip_axes=None):
    """Rigid motions D[k] (4x4), k = 0..e-g, taking the cube from its pose at the grasp to
    its pose at demo frame g+k - re-anchored at origin_now (where the cube is now).
      yaw_only  keep only the turn about vertical (drop the demo's tilt)
      end_xy    bend the path so the cube ends here (x-y), spread along the path
      floor_z   the cube's bottom never goes below this
      end_floor_z  stack2: the cube's bottom must END at least this high (the block's top +
                a few mm). If the demo's end is lower - its heights carry a few mm of camera
                error - the last half of the path is raised gradually so the cube doesn't
                hit the block early and get dragged short.
    Returns D, info."""
    S = smooth_segment(Tb[g:e + 1], times[g:e + 1])
    raw = Tb[g:e + 1]
    smooth_mm = 1000 * float(np.linalg.norm(S[:, :3, 3] - raw[:, :3, 3], axis=1).max())
    smooth_deg = max(rot_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(S, raw))
    Tg_inv = np.linalg.inv(S[0])
    shift = np.eye(4)
    shift[:3, 3] = origin_now - S[0, :3, 3]
    D = np.array([shift @ T @ Tg_inv @ np.linalg.inv(shift) for T in S])
    if yaw_only:
        for k in range(len(D)):
            o_t = D[k, :3, :3] @ origin_now + D[k, :3, 3]
            R = Rz(yaw_of(D[k, :3, :3]))
            D[k, :3, :3], D[k, :3, 3] = R, o_t - R @ origin_now
    o = np.array([D[k, :3, :3] @ origin_now + D[k, :3, 3] for k in range(len(D))])   # cube center path
    info = {"bend_m": 0.0, "floored": 0, "smooth_mm": smooth_mm, "smooth_deg": smooth_deg}
    if end_xy is not None:
        c = np.asarray(end_xy) - o[-1, :2]
        s = arc_fraction(o)
        D[:, :2, 3] += s[:, None] * c
        o[:, :2] += s[:, None] * c
        info["bend_m"] = float(np.linalg.norm(c))
    info["xy_scale"] = (XY_SCALE_X, XY_SCALE_Y)
    if (XY_SCALE_X, XY_SCALE_Y) != (1.0, 1.0):
        # stretch the x-y travel from the start, along the gripper's own x / y at the grasp
        ex = np.array([1.0, 0.0]) if grip_axes is None else grip_axes[0]
        ey = np.array([-ex[1], ex[0]]) if grip_axes is None else grip_axes[1]
        d = o[:, :2] - o[0, :2]
        stretch = (np.outer(d @ ex, ex) * (XY_SCALE_X - 1.0) + np.outer(d @ ey, ey) * (XY_SCALE_Y - 1.0))
        D[:, :2, 3] += stretch
        o[:, :2] += stretch
    if floor_z is not None:
        low = (o[:, 2] - half_height) < floor_z
        D[low, 2, 3] += floor_z - (o[low, 2] - half_height)
        o[low, 2] = floor_z + half_height
        info["floored"] = int(low.sum())
    info["end_raised_mm"] = 0.0
    if end_floor_z is not None:
        need = end_floor_z - (o[-1, 2] - half_height)
        if need > 0:
            s = arc_fraction(o)
            w = np.clip((s - 0.5) / 0.5, 0, 1)
            w = w * w * (3 - 2 * w)                          # smooth 0 -> 1 over the last half
            D[:, 2, 3] += need * w
            o[:, 2] += need * w
            info["end_raised_mm"] = 1000 * float(need)
    info["cube_path"] = o
    info["max_tilt_deg"] = max(rot_deg(Rz(-yaw_of(d[:3, :3])) @ d[:3, :3]) for d in D)
    info["end_yaw_deg"] = math.degrees(yaw_of(D[-1, :3, :3]))
    return D, info


def _corners(pts, tol):
    """Douglas-Peucker: indices of the points that make the path's shape (2D)."""
    if len(pts) < 3:
        return [0, len(pts) - 1]
    ab = pts[-1] - pts[0]
    L = np.linalg.norm(ab) + 1e-12
    d = np.abs(ab[0] * (pts[:, 1] - pts[0, 1]) - ab[1] * (pts[:, 0] - pts[0, 0])) / L
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [0, len(pts) - 1]
    return _corners(pts[:i + 1], tol)[:-1] + [i + j for j in _corners(pts[i:], tol)]


def push_segments(Tb, g, e, start_xy, times, turn_deg=45.0, step=0.005, min_m=0.015, corner_tol=0.015):
    """Split the demo's x-y path (re-anchored at start_xy) into LEGS at its sharp corners
    (where the hand had to come around to another face of the cube). An L-shaped demo
    gives 2 legs. Within a leg the pusher keeps ONE direction - the leg's overall
    direction - so it stays square behind the same face, while its position follows the
    demo path (gentle curves included).
    Returns [(points (n,2), directions (n,2) = the leg direction repeated, times (n,))]."""
    t = np.asarray(times[g:e + 1])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    pos = gauss(Tb[g:e + 1, :2, 3], SMOOTH_POS_S / dt)
    pos = pos - pos[0] + np.asarray(start_xy)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    n = max(3, int(s[-1] / step) + 1)
    ss = np.linspace(0, s[-1], n)
    pts = np.stack([np.interp(ss, s, pos[:, 0]), np.interp(ss, s, pos[:, 1])], 1)
    ts = np.interp(ss, s, t)
    # corners = shape points where the path turns by more than turn_deg
    c = _corners(pts, corner_tol)
    keep = [c[0]]
    for i in range(1, len(c) - 1):
        d1, d2 = pts[c[i]] - pts[keep[-1]], pts[c[i + 1]] - pts[c[i]]
        cosang = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12)
        if cosang < math.cos(math.radians(turn_deg)):
            keep.append(c[i])
    keep.append(c[-1])
    out = []
    for a, b in zip(keep[:-1], keep[1:]):
        if np.linalg.norm(pts[b] - pts[a]) < min_m:      # jitter while the cube sits still
            continue
        u = (pts[b] - pts[a]) / np.linalg.norm(pts[b] - pts[a])
        out.append((pts[a:b + 1], np.tile(u, (b - a + 1, 1)), ts[a:b + 1]))
    return out


def retarget_leg(seg, start_xy):
    """Re-place a leg so it starts where the cube really is (start_xy) but still ends where
    the demo's leg ends: the offset fades out along the leg. Returns the new leg + the
    offset at its start."""
    pts, u, ts = seg
    off = np.asarray(start_xy) - pts[0]
    f = arc_fraction(pts)[:, None]
    new = pts + (1 - f) * off
    v = (new[-1] - new[0]) / (np.linalg.norm(new[-1] - new[0]) + 1e-12)
    return (new, np.tile(v, (len(new), 1)), ts), off


# ----------------------------------------------------------- joint-space planning ---
def pick_knots(times):
    """Indices of the dense path to use as trajectory points: one every KNOT_S, plus the
    last (a too-short final gap is merged into the one before)."""
    t = np.asarray(times) - times[0]
    idx = [int(np.argmin(np.abs(t - x))) for x in np.arange(0, t[-1], KNOT_S)]
    idx = sorted(set(idx))
    if t[-1] - t[idx[-1]] < 0.4 * KNOT_S and len(idx) > 1:
        idx.pop()
    return idx + [len(t) - 1]


def _vel_acc(qs, dt):
    """Per-segment joint velocities and per-knot accelerations for segment durations dt."""
    d = dt[:, None]
    v = np.diff(qs, axis=0) / d
    span = np.vstack([d[:1], (d[1:] + d[:-1]) / 2, d[-1:]])
    return v, np.diff(np.vstack([np.zeros(6), v, np.zeros(6)]), axis=0) / span


def plan_path(poses, times, q_start, speed, accel, table_z, vmax=am.VMAX, amax=am.AMAX):
    """Turn a dense gripper path (4x4 poses at `times`) into a joint trajectory, starting
    from joint angles q_start (which must already be at poses[0]). Returns a dict:
      ok / why      passed every check, or the reasons it didn't
      q, t          joint angles and times (s from the start) of the trajectory points
      k, k_max      total time vs the demo (1 = demo timing), and the biggest local stretch
      + the numbers behind the checks, for the report."""
    idx = pick_knots(times)
    why, qs = [], [np.array(q_start, float)]
    for i in idx[1:]:
        q = am.ik(poses[i], qs[-1])
        if q is None:
            why.append(f"no IK at {times[i] - times[0]:.1f} s (out of reach / singular)")
            break
        qs.append(q)
    qs = np.array(qs)
    t = np.asarray(times)[idx[:len(qs)]] - times[0]
    p = {"ok": False, "why": why, "q": qs, "t": t, "k": 1.0, "n": len(qs),
         "demo_dur": float(times[-1] - times[0])}
    if why or len(qs) < 2:
        return p
    # timing: start from the demo's own timing. Segment velocity = change / time;
    # acceleration = change of velocity between neighbouring segments (the arm starts
    # and ends at rest). Wherever a joint would exceed its speed or acceleration limit,
    # lengthen just the segments around that spot, a little at a time, until every
    # segment is inside the limits. So the arm keeps the demo's timing where it can and
    # is slower only where the demo asks for more than the arm has.
    dt0 = np.diff(t)
    dt = dt0.copy()
    for _ in range(400):
        v, a = _vel_acc(qs, dt)
        vf = (np.abs(v) / (speed * vmax)).max(1)                 # per segment
        af = (np.abs(a) / (accel * amax)).max(1)                 # per knot (len = segments + 1)
        over = np.maximum(vf, np.maximum(af[:-1], af[1:]))
        if over.max() <= 1.0:
            break
        dt = dt * np.where(over > 1.0, 1.0 + 0.06 * np.minimum(over - 1.0, 1.0) + 0.01, 1.0)
    v, a = _vel_acc(qs, dt)
    t = np.r_[0, np.cumsum(dt)]
    k = float(t[-1] / dt0.sum())
    worst = int(np.unravel_index(np.argmax(np.abs(a) / amax), a.shape)[1]) + 1
    p.update(k=k, k_max=float((dt / dt0).max()), t=t, dur=float(t[-1]), a_joint=worst,
             v_peak=float((np.abs(v) / vmax).max()), a_peak=float((np.abs(a) / amax).max()),
             limits_ok=bool((np.abs(v) / (speed * vmax)).max() <= 1.001 and (np.abs(a) / (accel * amax)).max() <= 1.001))
    if not p["limits_ok"]:
        why.append("couldn't fit the motion inside the speed / acceleration limits")
    # checks
    p["margin_deg"] = min(am.limit_margin_deg(q) for q in qs)
    p["max_step"] = float(np.abs(np.diff(qs, axis=0)).max())
    p["body_low"] = min(float(f[2, 3]) for q in qs for f in am.frames(q)[2:6]) - table_z
    # path error vs the smoothed demo: joints interpolated linearly between the points
    # (the arm's own interpolation is at least this smooth), pushed through the arm model
    td = np.asarray(times) - times[0]
    t_demo = np.asarray(times)[idx] - times[0]                    # knot times before stretching
    dense = np.stack([np.interp(td, t_demo, qs[:, j]) for j in range(6)], 1)
    fkd = [am.fk(q) for q in dense]
    e_m = np.array([np.linalg.norm(f[:3, 3] - T[:3, 3]) for f, T in zip(fkd, poses)])
    e_d = np.array([rot_deg(f[:3, :3].T @ T[:3, :3]) for f, T in zip(fkd, poses)])
    p.update(err_cm_max=100 * float(e_m.max()), err_cm_mean=100 * float(e_m.mean()), err_deg_max=float(e_d.max()),
             path_m=float(np.sum(np.linalg.norm(np.diff([T[:3, 3] for T in poses], axis=0), axis=1))))
    if p["margin_deg"] < LIMIT_MARGIN_DEG:
        j = int(np.argmin((am.LIMIT - np.abs(qs)).min(0))) + 1
        why.append(f"joint_{j} gets within {p['margin_deg']:.0f} deg of its limit")
    if p["max_step"] > MAX_STEP_RAD:
        why.append(f"a joint jumps {math.degrees(p['max_step']):.0f} deg between two points (posture flip)")
    if p["body_low"] < BODY_CLEAR_M:
        why.append(f"wrist / elbow only {100 * p['body_low']:.0f} cm above the table")
    if p["err_cm_max"] > 100 * MAX_ERR_M or p["err_deg_max"] > MAX_ERR_DEG:
        why.append(f"path error {p['err_cm_max']:.1f} cm / {p['err_deg_max']:.1f} deg vs the demo")
    p["ok"] = not why
    return p


def postures(pose, q_now, n=40):
    """Different arm postures (IK branches) that reach `pose`, the one nearest q_now first.
    Deterministic: fixed random seeds."""
    rng = np.random.default_rng(0)
    out = []
    for seed in [np.asarray(q_now, float)] + [rng.uniform(-am.LIMIT, am.LIMIT) for _ in range(n)]:
        q = am.ik(pose, seed)
        if q is None or am.limit_margin_deg(q) < LIMIT_MARGIN_DEG:
            continue
        if all(np.abs(q - o).max() > 0.2 for o in out):
            out.append(q)
    return out


def descend(q_pre, pre, start, step_m=0.01):
    """Joint angles at `start`, reached from q_pre by a straight move from `pre` (the
    descent onto the cube / to pushing height), checked every 1 cm like the arm moves it.
    None if the arm would have to flip posture on the way."""
    n = max(1, int(math.ceil(np.linalg.norm(start[:3, 3] - pre[:3, 3]) / step_m)))
    q = np.asarray(q_pre, float)
    for s in np.linspace(0, 1, n + 1)[1:]:
        T = start.copy()
        T[:3, 3] = (1 - s) * pre[:3, 3] + s * start[:3, 3]
        q2 = am.ik(T, q)
        if q2 is None or np.abs(q2 - q).max() > 0.3 or am.limit_margin_deg(q2) < LIMIT_MARGIN_DEG:
            return None
        q = q2
    return q


def choose(options, q_now, speed, accel, table_z, vmax=am.VMAX, amax=am.AMAX):
    """options: list of (label, pre_pose, start_pose, make_path) where make_path() gives
    (poses, times) of the demo-following part starting at start_pose (the grasp / the
    push contact) and pre_pose is straight above it. Tries every option with every arm
    posture; returns the best plan that passes all checks (closest to the demo's timing,
    then furthest from the joint limits) or None, plus one line per option tried."""
    best, lines = None, []
    for label, pre, start, make_path in options:
        poses, times = make_path()
        tried = []
        for q_pre in postures(pre, q_now):
            q0 = descend(q_pre, pre, start)
            if q0 is None:
                continue
            p = plan_path(poses, times, q0, speed, accel, table_z, vmax, amax)
            p.update(label=label, q_pre=q_pre)
            tried.append(p)
            if p["ok"] and (best is None or (round(p["k"], 2), -p["margin_deg"]) <
                            (round(best["k"], 2), -best["margin_deg"])):
                best = p
        good = [p for p in tried if p["ok"]]
        if good:
            g = min(good, key=lambda p: p["k"])
            lines.append(f"{label}: ok ({len(good)}/{len(tried)} postures), x{g['k']:.2f} demo time")
        else:
            reason = tried[0]["why"][0] if tried and tried[0]["why"] else "no reachable posture"
            lines.append(f"{label}: no - {reason}")
    return best, lines


def report(p, extra=""):
    """The plan in a few lines (printed before anything moves, and logged)."""
    return (f"  plan [{p.get('label', '')}]: {p['n']} points over {p['dur']:.1f} s  (demo {p['demo_dur']:.1f} s, "
            f"x{p['k']:.2f}{', slowest part x%.1f' % p['k_max'] if p['k'] > 1.005 else ' = demo timing'})\n"
            f"    path {100 * p['path_m']:.1f} cm | error vs smoothed demo: max {p['err_cm_max']:.2f} cm, "
            f"mean {p['err_cm_mean']:.2f} cm, {p['err_deg_max']:.1f} deg\n"
            f"    peak joint speed {p['v_peak']:.0%} of limit, acceleration {p['a_peak']:.0%} (joint_{p['a_joint']}) | "
            f"closest to a joint limit {p['margin_deg']:.0f} deg | biggest step {math.degrees(p['max_step']):.0f} deg"
            + extra)


# ------------------------------------------------ what to try (shared with tests) ---
def down_pose(x, y, z, yaw_deg):
    """Tool pose pointing straight down, tool x-axis at yaw_deg about vertical
    (same convention as pick_place.down_with_yaw)."""
    a = math.radians(yaw_deg)
    zc, xc = np.array([0.0, 0.0, -1.0]), np.array([math.cos(a), math.sin(a), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([xc, np.cross(zc, xc), zc])
    T[:3, 3] = [x, y, z]
    return T


def carry_poses(Tb, times, g, e, top, geo, E_g, yaw_only, end_xy):
    """Dense gripper path that carries the cube along the demo, from grasp pose E_g.
    top = (x, y, z) of the cube's top-face center now. Returns poses, times, info."""
    origin = np.array([top[0], top[1], top[2] - geo["cube"] / 2])
    ax = E_g[:2, :2].T.copy()                              # gripper x and y axes, flattened onto the table
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-12
    D, info = carry_motion(Tb, times, g, e, origin, yaw_only=yaw_only, end_xy=end_xy,
                           floor_z=top[2] - geo["cube"], half_height=geo["cube"] / 2,
                           end_floor_z=geo.get("end_floor"), grip_axes=ax)
    return [d @ E_g for d in D], np.asarray(times[g:e + 1]), info


def grasp_heights(top_z, geo):
    """(tool z at the grasp, tool z above the cube) - same rule as pick_place.run()."""
    tips = max(top_z - geo["grip_depth"], top_z - geo["cube"] + geo["clear"])
    return tips + geo["tip"], top_z + geo["approach"] + geo["tip"]


def carry_options(Tb, times, g, e, top, geo, yaw0, end_xy=None, yaw_only=False):
    """Grasps to try: the cube's four equivalent angles (a cube looks the same turned
    90 deg), or only the ones in GRASP_YAWS (e.g. "180" or "180,0"). yaw_only=True follows only the demo's turn about vertical, not its tilt -
    the caller tries that only if no grasp can follow the full rotation."""
    gz, hz = grasp_heights(top[2], geo)
    opts, meta = [], {}
    yaws = ([float(v) for v in os.environ["GRASP_YAWS"].split(",")] if os.environ.get("GRASP_YAWS")
            else [(yaw0 + d) % 360 for d in (0, 90, -90, 180)])
    for yaw in yaws:
        yaw = yaw % 360
        label = f"grasp at {yaw:.0f} deg" + (", turn only (tilt dropped)" if yaw_only else "")
        start = down_pose(top[0], top[1], gz, yaw)
        pre = down_pose(top[0], top[1], hz, yaw)
        # TOOL_DX_M / TOOL_DY_M: the fingers' real center is off from tool_frame by this much,
        # in the GRIPPER's own x / y - so the correction turns with the grasp angle
        off = start[:3, :2] @ np.array([geo.get("tool_dx", 0.0), geo.get("tool_dy", 0.0)])
        start[:3, 3] -= off
        pre[:3, 3] -= off
        opts.append((label, pre, start,
                     lambda s=start: carry_poses(Tb, times, g, e, top, geo, s, yaw_only, end_xy)[:2]))
        meta[label] = (yaw, yaw_only)
    return opts, meta


def choose_carry(Tb, times, g, e, top, geo, yaw0, end_xy, q_now, speed, accel, yaw_only=False, **lim):
    """Best grasp + posture for carrying the cube along the demo. Full rotation first;
    turn-only as the fallback. Returns plan (or None), (yaw, yaw_only), lines tried."""
    lines = []
    for yo in ([True] if yaw_only else [False, True]):
        opts, meta = carry_options(Tb, times, g, e, top, geo, yaw0, end_xy, yo)
        best, ln = choose(opts, q_now, speed, accel, top[2] - geo["cube"], **lim)
        lines += ln
        if best is not None:
            return best, meta[best["label"]], lines
    return None, None, lines


def push_options(seg, top_z, geo):
    """One push (points, directions, times from push_segments): the closed gripper slides
    just behind the cube along the demo path. A closed gripper is the same turned 180 deg,
    so both are tried."""
    pts, u, ts = seg
    low = top_z - geo["cube"] + geo["push_low"] + geo["tip"]
    safe = top_z + geo["approach"] + geo["tip"]
    contact = pts - u * (geo["cube"] / 2 + 0.003)
    yaw0 = math.degrees(math.atan2(*u.mean(0)[::-1])) + geo["push_yaw_off"]
    opts, meta = [], {}
    for d in (0, 180):
        yaw = (yaw0 + d) % 360
        label = f"pusher at {yaw:.0f} deg"
        behind = contact[0] - u[0] * 0.03
        lead = [down_pose(behind[0], behind[1], low, yaw)] + [down_pose(c[0], c[1], low, yaw) for c in contact]
        lt = np.r_[ts[0] - 0.5, ts]                       # 0.5 s to close the 3 cm gap
        opts.append((label, down_pose(behind[0], behind[1], safe, yaw), lead[0], lambda l=lead, t=lt: (l, t)))
        meta[label] = (yaw, behind, low, safe, contact[-1] - u[-1] * 0.0, u[-1])
    return opts, meta