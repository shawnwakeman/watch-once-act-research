#!/usr/bin/env python3
"""Stage 1 of the pipeline on one run: find the object, mask it, track points on it.

  Grounding DINO (frame 0, text prompt) -> box
  SAM (frame 0, box)                     -> mask
  sample points inside the mask          -> queries
  CoTracker3 online (short windows)      -> 2D track + visibility for every point, every frame

Usage (Mac, in the perception venv):
  python3 perceive.py runs/pick_place/run01 --prompt "blue block"
  python3 perceive.py runs/bowl_drop/run02 --prompt "rubber duck" --target "white box"
  python3 perceive.py runs/pick_place/run01 --prompt "blue block" --source phone   # a video level

Writes into the run folder, under perception/<source>/:
  detections.json   box (+ score) for the object (and the target, if given)
  mask.png          SAM mask of the object in frame 0 (target_mask.png for the target)
  tracks.npz        tracks (T, N, 2) pixel x,y; visible (T, N) bool; queries (N, 2); size
  overlay.mp4       the clip with the box, mask outline and tracked points drawn - watch it!

Models download from Hugging Face / torch.hub on first use (~1 GB total), then are cached.
Runs on Apple's GPU (MPS) when it can, otherwise the CPU.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"


def pick_device(want):
    if want != "auto":
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frames(run_dir, source, max_side):
    """All frames of the run as RGB uint8 (T, H, W, 3), downscaled so the long side <= max_side."""
    if source == "xtion":
        files = sorted((run_dir / "xtion" / "rgb").glob("*.png"))
        frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    else:
        cap = cv2.VideoCapture(str(run_dir / f"{source}.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not frames:
        raise SystemExit(f"No frames for source '{source}' in {run_dir}")
    h, w = frames[0].shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frames = [cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) for f in frames]
    return np.stack(frames), scale, (w, h)


def detect(image, prompt, device, box_thr=0.30, text_thr=0.25):
    """Grounding DINO: the highest-scoring box for the prompt in one image."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(DINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(device).eval()
    text = prompt.lower().strip().rstrip(".") + "."          # DINO wants lowercase, ending in '.'
    pil = Image.fromarray(image)
    inputs = proc(images=pil, text=text, return_tensors="pt")
    inputs = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in inputs.items()}
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    kw = dict(text_threshold=text_thr, target_sizes=[pil.size[::-1]])
    try:   # the argument name changed between transformers versions
        res = proc.post_process_grounded_object_detection(out, inputs["input_ids"], threshold=box_thr, **kw)[0]
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inputs["input_ids"], box_threshold=box_thr, **kw)[0]
    if len(res["scores"]) == 0:
        return None
    i = int(torch.argmax(res["scores"]))
    return {"box": [float(v) for v in res["boxes"][i].tolist()], "score": float(res["scores"][i]),
            "n_candidates": int(len(res["scores"]))}


def segment(image, box, device):
    """SAM: the mask of the object inside the box."""
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(SAM_ID)
    model = SamModel.from_pretrained(SAM_ID).to(device).eval()
    inputs = proc(Image.fromarray(image), input_boxes=[[box]], return_tensors="pt")
    # Apple's GPU (MPS) has no float64; the processor makes the box tensor float64
    inputs = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()},
                    multimask_output=False)
    masks = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"],
                                                    inputs["reshaped_input_sizes"])
    return masks[0][0, 0].numpy().astype(bool)


def sample_points(mask, n, erode_px=3):
    """About n points spread evenly inside the mask, kept a few px away from its edge
    (edge pixels mix object and background, and their depth is unreliable)."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8))
    if m.sum() < n:
        m = mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    step = max(1, int(np.sqrt(len(xs) / n)))
    keep = (xs % step == 0) & (ys % step == 0)
    pts = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    if len(pts) > n:
        pts = pts[np.linspace(0, len(pts) - 1, n).astype(int)]
    return pts


def track(frames, queries, device):
    """CoTracker3 ONLINE: every query point through the clip, a short window at a time.
    (The offline model holds the whole clip in memory at once - too much for a laptop.)
    Only the current window is on the GPU; the clip itself stays in normal memory."""
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    T = len(frames)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()              # B T C H W, 0-255
    q = torch.cat([torch.zeros(len(queries), 1), torch.from_numpy(queries)], 1)[None]  # (t=0, x, y)
    for dev in ([device, torch.device("cpu")] if device.type != "cpu" else [device]):
        try:
            model = model.to(dev).eval()
            step = model.step
            # pad the end by repeating the last frame so the final windows cover every frame
            pad = video[:, -1:].expand(-1, 2 * step, -1, -1, -1)
            vid = torch.cat([video, pad], 1)
            tr = vis = None
            with torch.no_grad():
                model(video_chunk=vid[:, :2 * step].to(dev), is_first_step=True, queries=q.to(dev))
                for i in range(0, vid.shape[1] - step, step):
                    tr, vis = model(video_chunk=vid[:, i:i + 2 * step].to(dev))
                    if (i // step) % 10 == 0:
                        print(f"    tracked to frame {min(i + 2 * step, T)}/{T}", flush=True)
            tr, vis = tr[0, :T].cpu().numpy(), vis[0, :T].cpu().numpy()
            if vis.ndim == 3:
                vis = vis[..., 0]
            return tr, vis.astype(bool) if vis.dtype == bool else (vis > 0.5)
        except (RuntimeError, NotImplementedError) as e:
            if dev.type == "cpu":
                raise
            print(f"  CoTracker failed on {dev} ({str(e)[:80]}...) - retrying on the CPU (slower)")
            if dev.type == "mps":
                torch.mps.empty_cache()
    raise RuntimeError("unreachable")


def draw_overlay(frames, tracks, visible, box, mask, path, fps=30):
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rng = np.random.default_rng(0)
    colors = rng.integers(60, 255, (tracks.shape[1], 3)).tolist()
    for t, f in enumerate(frames):
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        if t == 0:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
        for i in range(tracks.shape[1]):
            x, y = tracks[t, i]
            if visible[t, i]:
                cv2.circle(img, (int(x), int(y)), 2, colors[i], -1)
            else:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), 1)   # red ring = hidden
        cv2.putText(img, f"frame {t}  visible {int(visible[t].sum())}/{tracks.shape[1]}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(img)
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", required=True, help='what to track, e.g. "blue block"')
    ap.add_argument("--target", help='two-object tasks: the target, e.g. "white box"')
    ap.add_argument("--source", default="xtion", help="xtion (L0) or phone / ipad / mac (video levels)")
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--max-side", type=int, default=640, help="downscale frames so the long side <= this")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    out = run_dir / "perception" / a.source
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(a.device)
    print(f"{run_dir} [{a.source}] on {device}")

    frames, scale, orig_size = load_frames(run_dir, a.source, a.max_side)
    print(f"  {len(frames)} frames, {frames.shape[2]}x{frames.shape[1]} (scale {scale:.2f} of {orig_size[0]}x{orig_size[1]})")

    det = detect(frames[0], a.prompt, device)
    if det is None:
        raise SystemExit(f"  Grounding DINO found no '{a.prompt}' in frame 0 - try another prompt "
                         "(color + object, e.g. 'red cube'), or this is a detection failure for this level.")
    print(f"  DINO: '{a.prompt}' score {det['score']:.2f} box {[round(v) for v in det['box']]}")
    dets = {"object": {"prompt": a.prompt, **det}}
    mask = segment(frames[0], det["box"], device)
    cv2.imwrite(str(out / "mask.png"), mask.astype(np.uint8) * 255)
    print(f"  SAM: mask {int(mask.sum())} px")

    if a.target:
        tdet = detect(frames[0], a.target, device)
        if tdet is None:
            print(f"  !! target '{a.target}' not found in frame 0")
        else:
            dets["target"] = {"prompt": a.target, **tdet}
            tm = segment(frames[0], tdet["box"], device).astype(np.uint8) * 255
            cv2.imwrite(str(out / "target_mask.png"), cv2.resize(tm, orig_size, interpolation=cv2.INTER_NEAREST))
            print(f"  DINO target: score {tdet['score']:.2f}")

    queries = sample_points(mask, a.points)
    print(f"  tracking {len(queries)} points with CoTracker3 (online, windowed) ...")
    tracks, visible = track(frames, queries, device)

    # store everything in ORIGINAL pixel coordinates (that's what the intrinsics K refer to)
    inv = 1.0 / scale
    np.savez_compressed(out / "tracks.npz", tracks=tracks * inv, visible=visible, queries=queries * inv,
                        size=np.array(orig_size), scale_used=scale)
    for d in dets.values():
        d["box"] = [v * inv for v in d["box"]]
    (out / "detections.json").write_text(json.dumps(dets, indent=2))
    draw_overlay(frames, tracks, visible, [v * scale for v in dets["object"]["box"]], mask, out / "overlay.mp4")
    if mask.shape != tuple(orig_size[::-1]):
        cv2.imwrite(str(out / "mask.png"), cv2.resize(mask.astype(np.uint8) * 255, orig_size, interpolation=cv2.INTER_NEAREST))
    vis_frac = visible.mean(1)
    print(f"  visible points: {vis_frac.mean():.0%} on average, lowest {vis_frac.min():.0%} (frame {int(vis_frac.argmin())})")
    print(f"  -> {out}/  (watch overlay.mp4)")


if __name__ == "__main__":
    main()