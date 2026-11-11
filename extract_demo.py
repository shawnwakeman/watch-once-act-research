#!/usr/bin/env python3
"""Export a demo bag to plain files for the perception pipeline (no ROS needed after).

    python3 extract_demo.py /root/kinova_ws/demos/cube_a_to_b/cube_a_to_b_take01_....bag

Writes next to the bag, in a folder with the same name:
    rgb/000000.png ...     color frames (8-bit BGR, 640x480)
    depth/000000.png ...   registered depth, 16-bit PNG in millimeters (0 = no reading),
                           matched to each color frame by nearest timestamp
    frames.csv             index, rgb_time, depth_time, gap_ms
    camera.yaml            K, D, P, size (from camera_info) + base_link->camera_link if recorded
    rgb.mp4                a quick-look video (if ffmpeg is installed)
"""
import bisect
import os
import subprocess
import sys

import cv2
import numpy as np
import rosbag
import yaml
from cv_bridge import CvBridge

RGB = "/camera/rgb/image_raw"
DEPTH = "/camera/depth_registered/image_raw"
INFO = "/camera/rgb/camera_info"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bag_path = sys.argv[1]
    out = os.path.splitext(bag_path)[0]
    os.makedirs(f"{out}/rgb", exist_ok=True)
    os.makedirs(f"{out}/depth", exist_ok=True)
    br = CvBridge()

    depth_t, depth_imgs, info, calib = [], [], None, None
    bag = rosbag.Bag(bag_path)
    for topic, msg, _ in bag.read_messages(topics=[DEPTH, INFO, "/tf_static"]):
        if topic == DEPTH:
            d = br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if d.dtype != np.uint16:                       # 32FC1 meters -> uint16 mm
                d = np.nan_to_num(d * 1000.0).astype(np.uint16)
            depth_t.append(msg.header.stamp.to_sec())
            depth_imgs.append(d)
        elif topic == INFO and info is None:
            info = msg
        elif topic == "/tf_static":
            for tr in msg.transforms:
                if tr.header.frame_id == "base_link" and tr.child_frame_id == "camera_link":
                    t, r = tr.transform.translation, tr.transform.rotation
                    calib = {"x": t.x, "y": t.y, "z": t.z, "qx": r.x, "qy": r.y, "qz": r.z, "qw": r.w}

    rows, i = [], 0
    for _, msg, _ in bag.read_messages(topics=[RGB]):
        ts = msg.header.stamp.to_sec()
        cv2.imwrite(f"{out}/rgb/{i:06d}.png", br.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
        gap = None
        if depth_t:
            j = bisect.bisect_left(depth_t, ts)
            j = min([k for k in (j - 1, j) if 0 <= k < len(depth_t)], key=lambda k: abs(depth_t[k] - ts))
            cv2.imwrite(f"{out}/depth/{i:06d}.png", depth_imgs[j])
            gap = 1000 * abs(depth_t[j] - ts)
        rows.append(f"{i},{ts:.6f},{depth_t[j] if depth_t else ''},{'' if gap is None else f'{gap:.1f}'}")
        i += 1
    bag.close()

    with open(f"{out}/frames.csv", "w") as f:
        f.write("index,rgb_time,depth_time,gap_ms\n" + "\n".join(rows) + "\n")
    cam = {"source_bag": os.path.basename(bag_path), "frames": i}
    if info:
        cam.update(width=info.width, height=info.height, K=list(info.K), D=list(info.D), P=list(info.P),
                   distortion_model=info.distortion_model)
    if calib:
        cam["base_link_to_camera_link"] = calib
    with open(f"{out}/camera.yaml", "w") as f:
        yaml.safe_dump(cam, f, sort_keys=False)

    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", f"{out}/rgb/%06d.png",
                        "-pix_fmt", "yuv420p", f"{out}/rgb.mp4"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    gaps = [float(r.split(",")[3]) for r in rows if r.split(",")[3]]
    print(f"{i} color frames, {len(depth_t)} depth frames -> {out}/")
    if gaps:
        print(f"depth-to-color time gap: median {np.median(gaps):.1f} ms, max {max(gaps):.1f} ms")
    if not info:
        print("WARNING: no camera_info in the bag")


if __name__ == "__main__":
    main()