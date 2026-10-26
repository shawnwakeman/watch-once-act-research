#!/usr/bin/env python3
"""ArUco detector for OpenCV dictionaries (e.g. chev.me "4x4" markers), as a stand-in for
aruco_ros, which only knows the "Original ArUco" family.

Publishes the marker's pose to tf as  camera_rgb_optical_frame -> calib_marker,
and an annotated image on /aruco_single/result (outline + axes when detected).

Normally started by calibrate.launch (dictionary:=DICT_4X4_1000). Standalone:
    python3 aruco_cv_node.py _marker_id:=0 _marker_size:=0.1 _dictionary:=DICT_4X4_1000
"""
from collections import deque

import numpy as np
import cv2
import rospy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image


# "Steady" = over the averaging window, position wobble and worst rotation
# deviation are below these. The banner on the image turns green when steady.
STEADY_POS_MM = 2.0
STEADY_ROT_DEG = 1.5


def rot_angle(R1, R2):
    """Angle in degrees between two rotations."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rotmat_to_quat(R):
    """3x3 rotation matrix -> (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


class ArucoNode:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise SystemExit("This OpenCV has no aruco module. Run:  pip3 install opencv-contrib-python==4.6.0.66")
        self.marker_id = int(rospy.get_param("~marker_id", 0))
        self.size = float(rospy.get_param("~marker_size", 0.1))
        dict_name = rospy.get_param("~dictionary", "DICT_4X4_1000")
        self.frame = rospy.get_param("~marker_frame", "calib_marker")

        d = getattr(cv2.aruco, dict_name)
        # OpenCV < 4.7 and >= 4.7 have different aruco APIs - support both.
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(d)
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            det = cv2.aruco.ArucoDetector(self.dictionary, params)
            self.detect = lambda g: det.detectMarkers(g)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(d)
            params = cv2.aruco.DetectorParameters_create()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detect = lambda g: cv2.aruco.detectMarkers(g, self.dictionary, parameters=params)

        # Marker corners in its own frame (z out of the marker face), for solvePnP
        h = self.size / 2.0
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

        self.K = None
        self.hist = deque(maxlen=int(rospy.get_param("~average_frames", 15)))  # ~0.5 s at 30 fps
        self.last_seen = None
        self.bridge = CvBridge()
        self.tf = tf2_ros.TransformBroadcaster()
        self.pub = rospy.Publisher("/aruco_single/result", Image, queue_size=1)
        rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.on_info, queue_size=1)
        rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.on_image, queue_size=1, buff_size=2**24)
        rospy.loginfo("aruco_cv_node: %s, id %d, size %.3f m", dict_name, self.marker_id, self.size)

    def on_info(self, msg):
        # Image is rectified, so use the projection matrix P and no distortion
        self.K = np.array(msg.P, dtype=np.float64).reshape(3, 4)[:, :3]

    # --- pose estimation helpers -------------------------------------------------

    def solve_pose(self, img_pts):
        """Marker pose (R, t). IPPE gives two candidate poses for a square; a nearly
        face-on or small marker makes them close, which shows up as the axes 'flipping'.
        Pick the best-fitting one, and if both fit about equally, the one closest to the
        recent average so the pose can't jump between them."""
        D = np.zeros(5)
        if hasattr(cv2, "solvePnPGeneric"):
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.obj, img_pts, self.K, D,
                                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
            cands = [(cv2.Rodrigues(r)[0], t.flatten(), float(e)) for r, t, e in
                     zip(rvecs, tvecs, np.array(errs).flatten())]
            cands.sort(key=lambda c: c[2])
            if len(cands) > 1 and self.hist and cands[1][2] < 1.5 * cands[0][2] + 0.5:
                R_ref = self.mean_pose()[0]
                cands.sort(key=lambda c: rot_angle(c[0], R_ref))
            return cands[0][0], cands[0][1]
        ok, rvec, tvec = cv2.solvePnP(self.obj, img_pts, self.K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return cv2.Rodrigues(rvec)[0], tvec.flatten()

    def mean_pose(self):
        Rs = np.array([h[0] for h in self.hist])
        ts = np.array([h[1] for h in self.hist])
        U, _, Vt = np.linalg.svd(Rs.mean(axis=0))      # average rotation, projected back to a rotation
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R, ts.mean(axis=0), ts, Rs

    def on_image(self, msg):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect(gray)
        now = rospy.Time.now()

        found = False
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            for c, i in zip(corners, ids.flatten()):
                if i != self.marker_id:
                    continue
                R, t = self.solve_pose(c.reshape(4, 2).astype(np.float32))
                self.hist.append((R, t))
                self.last_seen = now
                found = True
                break

        # Forget the history if the marker was lost for a moment (arm probably moved)
        if not found and self.last_seen is not None and (now - self.last_seen).to_sec() > 0.5:
            self.hist.clear()

        status, color = "NO MARKER", (0, 0, 255)
        if len(self.hist) >= 5:
            R, t, ts, Rs = self.mean_pose()
            pos_sd_mm = 1000 * np.linalg.norm(ts.std(axis=0))
            rot_sd_deg = max(rot_angle(Ri, R) for Ri in Rs)
            steady = (len(self.hist) == self.hist.maxlen and pos_sd_mm < STEADY_POS_MM
                      and rot_sd_deg < STEADY_ROT_DEG)
            status = "%s  pos %.1f mm  rot %.1f deg  (%d frames)" % (
                "STEADY - OK TO SAMPLE" if steady else "WAIT - not steady", pos_sd_mm, rot_sd_deg, len(self.hist))
            color = (0, 200, 0) if steady else (0, 165, 255)

            rvec, _ = cv2.Rodrigues(R)
            cv2.drawFrameAxes(img, self.K, np.zeros(5), rvec, t, self.size * 0.5)
            qx, qy, qz, qw = rotmat_to_quat(R)
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = msg.header.frame_id     # camera_rgb_optical_frame
            tf.child_frame_id = self.frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = t
            tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            self.tf.sendTransform(tf)    # averaged pose over the last frames
        elif found:
            status, color = "WAIT - collecting frames", (0, 165, 255)

        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        out.header = msg.header
        self.pub.publish(out)


if __name__ == "__main__":
    rospy.init_node("aruco_cv_node")
    ArucoNode()
    rospy.spin()