"""Grounding DINO + SAM on the robot's live camera image (replay.py --dino).

The SAME models and the same steps as the Mac pipeline (perception/perceive.py):
  text prompt -> Grounding DINO -> best box -> SAM -> mask of the object
replay.py then turns the mask into 3D with the point cloud.

Install once INSIDE the container (CPU is fine: ~2-5 s per image). Installs go into
kinova_ws (mounted from the lab PC), so they survive the container being recreated:
  pip3 install --target /root/kinova_ws/pylibs torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
  pip3 install --target /root/kinova_ws/pylibs "transformers==4.44.2" pillow
The models (~800 MB) download on first use into /root/kinova_ws/hf_cache.
"""
import os
import sys

import cv2
import numpy as np

_WS = "/root/kinova_ws"
if os.path.isdir(os.path.join(_WS, "pylibs")) and os.path.join(_WS, "pylibs") not in sys.path:
    sys.path.insert(0, os.path.join(_WS, "pylibs"))
if os.path.isdir(_WS):
    os.environ.setdefault("HF_HOME", os.path.join(_WS, "hf_cache"))

DINO_ID = "IDEA-Research/grounding-dino-tiny"         # same as perceive.py
SAM_ID = "facebook/sam-vit-base"


class Detector:
    """Loads the models once (the first call takes ~20-60 s), then find() per image."""

    def __init__(self):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        except ImportError as e:
            raise SystemExit(f"--dino needs torch + transformers in the container ({e}). See live_detect.py "
                             "for the two pip commands.")
        self.torch = torch
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  loading Grounding DINO + SAM on {self.dev} (first time: download ~800 MB) ...")
        self.dproc = AutoProcessor.from_pretrained(DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(self.dev).eval()
        self.sproc = SamProcessor.from_pretrained(SAM_ID)
        self.sam = SamModel.from_pretrained(SAM_ID).to(self.dev).eval()

    def _to(self, inputs):
        t = self.torch
        return {k: ((v.float() if v.dtype == t.float64 else v).to(self.dev) if t.is_tensor(v) else v)
                for k, v in inputs.items()}

    def boxes(self, rgb, prompt, box_thr=0.25, text_thr=0.25):
        """All DINO boxes for the prompt, best first: [(score, [x0, y0, x1, y1])]."""
        from PIL import Image
        pil = Image.fromarray(rgb)
        text = prompt.lower().strip().rstrip(".") + "."            # DINO wants lowercase ending in '.'
        inputs = self._to(self.dproc(images=pil, text=text, return_tensors="pt"))
        with self.torch.no_grad():
            out = self.dino(**inputs)
        kw = dict(text_threshold=text_thr, target_sizes=[pil.size[::-1]])
        try:                                     # the argument name changed between transformers versions
            res = self.dproc.post_process_grounded_object_detection(out, inputs["input_ids"], threshold=box_thr, **kw)[0]
        except TypeError:
            res = self.dproc.post_process_grounded_object_detection(out, inputs["input_ids"], box_threshold=box_thr, **kw)[0]
        found = [(float(s), [float(v) for v in b]) for s, b in zip(res["scores"].tolist(), res["boxes"].tolist())]
        return sorted(found, key=lambda x: -x[0])

    def mask(self, rgb, box):
        """SAM mask (H, W) bool of the object inside the box."""
        from PIL import Image
        inputs = self.sproc(Image.fromarray(rgb), input_boxes=[[box]], return_tensors="pt")
        orig, resh = inputs["original_sizes"], inputs["reshaped_input_sizes"]
        with self.torch.no_grad():
            out = self.sam(**self._to(inputs), multimask_output=False)
        m = self.sproc.image_processor.post_process_masks(out.pred_masks.cpu(), orig, resh)
        return m[0][0, 0].numpy().astype(bool)

    def find(self, bgr, prompt, avoid=(), name="obj"):
        """Best box for `prompt` that doesn't overlap any box in `avoid` (IoU > 0.3), and its
        SAM mask. Saves /tmp/replay_dino_<name>.png. Returns (mask, box, score) or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cands = self.boxes(rgb, prompt)
        pick = next(((s, b) for s, b in cands if all(iou(b, a) < 0.3 for a in avoid)), None)
        dbg = bgr.copy()
        for s, b in cands:
            x0, y0, x1, y1 = (int(v) for v in b)
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0) if (s, b) == pick else (0, 0, 255), 2)
            cv2.putText(dbg, f"{s:.2f}", (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if pick is None:
            cv2.imwrite(f"/tmp/replay_dino_{name}.png", dbg)
            return None
        m = self.mask(rgb, pick[1])
        dbg[m] = (0.5 * dbg[m] + [0, 127, 0]).astype(np.uint8)
        cv2.imwrite(f"/tmp/replay_dino_{name}.png", dbg)
        return m, pick[1], pick[0]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)