"""Kinova Gen3 Lite kinematics in plain numpy (no ROS), from Kinova's own URDF
(ros_kortex: gen3_lite_macro.xacro + gen3_lite_2f_macro.xacro). Used to plan and CHECK a
whole demo-following motion in joint space before the arm touches anything, and to run
the same check offline (test_follow.py).

On the robot, replay.py compares fk(current joints) with MoveIt's tool_frame pose at
start-up and refuses to follow if they disagree - so a wrong model can't move the arm.
"""
import math

import numpy as np

# joint origins (xyz, rpy) parent->child; every joint turns about its own z axis
_J = [((0, 0, 0.12825), (0, 0, 0)),
      ((0, -0.03, 0.115), (1.5708, 0, 0)),
      ((0, 0.28, 0), (-3.1416, 0, 0)),
      ((0, -0.14, 0.02), (1.5708, 0, 0)),
      ((0.0285, 0, 0.105), (0, 1.5708, 0)),
      ((-0.105, 0, 0.0285), (0, -1.5708, 0))]
_TOOL = ((0, 0, 0.130), (0, 0, math.pi / 2))             # end_effector_link -> tool_frame

LIMIT = np.array([2.68, 2.61, 2.61, 2.60, 2.53, 2.60])   # +- rad (URDF)
# MoveIt's limits for this arm (kortex_move_it_config joint_limits.yaml); replay.py
# replaces them with the live ones from the parameter server when it can.
VMAX = np.array([0.48, 0.48, 0.48, 0.76, 0.76, 1.52])    # rad/s
AMAX = np.array([0.86, 0.43, 0.34, 0.86, 8.6, 8.6])      # rad/s^2


def _T(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = xyz
    return T


_FIX = [_T(*j) for j in _J]
_TOOL_T = _T(*_TOOL)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def frames(q):
    """Pose of every joint frame (after its rotation) and of tool_frame, in base_link."""
    out, T = [], np.eye(4)
    for fix, a in zip(_FIX, q):
        T = T @ fix @ _rz(a)
        out.append(T)
    out.append(T @ _TOOL_T)
    return out


def fk(q):
    """tool_frame pose (4x4) in base_link."""
    return frames(q)[-1]


def rot_log(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    th = math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:                               # near 180 deg: axis from the diagonal
        ax = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        ax[1] = math.copysign(ax[1], R[0, 1]) if ax[0] > 1e-6 else ax[1]
        ax[2] = math.copysign(ax[2], R[0, 2]) if ax[0] > 1e-6 else math.copysign(ax[2], R[1, 2])
        return ax * th
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * th / (2 * math.sin(th))


def jacobian(q):
    fr = frames(q)
    p = fr[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = fr[i][:3, 2], fr[i][:3, 3]
        J[:3, i], J[3:, i] = np.cross(z, p - o), z
    return J


def ik(T, seed, iters=150, tol_m=1e-4, tol_rad=1e-3):
    """Joint angles reaching tool pose T, found by stepping from `seed` (damped least
    squares) - so it returns the solution on the same 'branch' as the seed, which is what
    keeps a path continuous. None if it doesn't converge."""
    q = np.array(seed, float)
    for _ in range(iters):
        cur = fk(q)
        e = np.r_[T[:3, 3] - cur[:3, 3], rot_log(T[:3, :3] @ cur[:3, :3].T)]
        if np.linalg.norm(e[:3]) < tol_m and np.linalg.norm(e[3:]) < tol_rad:
            return (q + math.pi) % (2 * math.pi) - math.pi
        J = jacobian(q)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        m = np.abs(dq).max()
        q = q + (dq if m < 0.3 else dq * 0.3 / m)
    return None


def limit_margin_deg(q):
    """Smallest distance of any joint to its limit, degrees (negative = outside)."""
    return float(np.degrees((LIMIT - np.abs(q)).min()))