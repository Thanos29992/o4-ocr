#!/usr/bin/env python3
"""
PP-OCRv4 (Intel/ocr-text-recognition) OCR pipeline over OpenVINO.

Runs the full text-detection + text-recognition pipeline on CPU / GPU / NPU
(any OpenVINO device). This is the model family behind Intel's
`Intel/ocr-text-recognition` HF repo, converted to OpenVINO IR.

Usage:
    python ocr_pipeline.py --image <path> [--device CPU|NPU|GPU|AUTO] \
        [--det-shape WxH] [--rec-width N] [--dir models/ppocrv4]

The detection model gets a fixed square canvas (default 960x960, padded).
The recognition model has dynamic width, which NPU does not accept, so for
NPU we pad every crop to a fixed width (`--rec-width`, default 320) and
feed the padded blob; the CTC decoder naturally ignores padded trailing
blank frames. On CPU/GPU the rec width stays dynamic.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov


# --------------------------------------------------------------------------
# Model / dict layout
# --------------------------------------------------------------------------
def load_models(model_dir: Path, device: str,
                det_shape=(960, 960), rec_fixed_width=None):
    """Load det + rec IR and compile for `device`.

    Returns (core, det_compiled, rec_compiled, dict_character).
    """
    core = ov.Core()
    det_xml = model_dir / "ch_PP-OCRv4_det_infer" / "inference.xml"
    rec_xml = model_dir / "ch_PP-OCRv4_rec_server_infer" / "inference.xml"
    dict_file = model_dir / "ppocr_keys_v1.txt"

    # -- detection -------------------------------------------------------
    det = core.read_model(det_xml)
    # NPU needs fully static shapes: pin to a fixed square canvas.
    drh, drw = det_shape
    det.reshape({det.input(0).any_name: ov.PartialShape([1, 3, drh, drw])})
    det_c = core.compile_model(det, device)

    # -- recognition -----------------------------------------------------
    rec = core.read_model(rec_xml)
    rec_in = rec.input(0).any_name
    if rec_fixed_width is not None:
        # NPU requires static width too -> pad crops to a fixed width.
        rec.reshape({rec_in: ov.PartialShape([1, 3, 48, rec_fixed_width])})
        rec_c = core.compile_model(rec, device)
    else:
        # CPU/GPU tolerate dynamic width -> keep it, avoid resizing loss.
        for il in rec.inputs:
            pshape = il.get_partial_shape()
            pshape[3] = ov.Dimension(-1)
            rec.reshape({il: pshape})
        rec_c = core.compile_model(rec, device)

    # -- char dict for CTC decode ----------------------------------------
    chars = open(dict_file, encoding="utf-8").read().splitlines()
    dict_character = ["blank"] + chars + [" "]

    return core, det_c, rec_c, dict_character


# --------------------------------------------------------------------------
# Pre / post-processing
# --------------------------------------------------------------------------
def detect_text_regions(det_c, det_out_idx, img, det_shape=(960, 960),
                        thresh=0.3, pad=4, min_w=10, min_h=5):
    """Return list of (x, y, w, h) text boxes in original-image coords."""
    h0, w0 = img.shape[:2]
    drh, drw = det_shape
    scale = drh / max(h0, w0)                # fit into square canvas
    resized = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
    canvas = np.zeros((drh, drw, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized

    blob = canvas.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    det_map = det_c([blob])[det_out_idx][0, 0]
    binary = (det_map > thresh).astype(np.uint8) * 255

    # Dilate so a whole word merges into one box instead of per-character.
    dilated = cv2.dilate(binary, np.ones((9, 25), np.uint8))
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        x0, y0 = max(0, x / scale - pad), max(0, y / scale - pad)
        x1, y1 = min(w0, (x + w) / scale + pad), min(h0, (y + h) / scale + pad)
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
    # Left-to-right, top-to-bottom reading order.
    boxes.sort(key=lambda b: (b[1] // 20, b[0]))
    return boxes


def ctc_decode(preds, dict_character):
    """preds: [T, num_classes] softmax probabilities -> text string."""
    idx = preds.argmax(axis=1)
    out = []
    prev = -1
    for i in idx:
        if i != prev and i != 0:           # skip blank (0) and repeats
            out.append(dict_character[i])
        prev = i
    return "".join(out)


def recognize_text(rec_c, rec_out_idx, crop, dict_character,
                   rec_h=48, max_w=None, dyn_cap=960):
    """CRNN-CTC recognize one cropped text region -> (text, mean_conf).

    max_w is the FIXED input width for NPU-style static shapes (crops are
    padded with blank frames to fit; trailing blanks decode to nothing). When
    max_w is None the model has a dynamic width axis (CPU/GPU) and each crop
    is resized to its natural aspect-ratio width, bounded by `dyn_cap` to
    avoid pathological extremes.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return "", 0.0
    if max_w is None:                      # dynamic width (CPU/GPU)
        resized_w = max(1, min(dyn_cap, round(rec_h * w / h)))
        blob = cv2.resize(crop, (resized_w, rec_h))
    else:                                  # fixed width (NPU) -> pad
        resized_w = max(1, min(max_w, round(rec_h * w / h)))
        resized = cv2.resize(crop, (resized_w, rec_h))
        blob = np.zeros((rec_h, max_w, 3), dtype=np.uint8)
        blob[:, :resized_w] = resized

    blob = blob.astype(np.float32)
    blob = ((blob - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis, ...]

    preds = np.asarray(rec_c([blob])[rec_out_idx][0])    # [T, C]
    text = ctc_decode(preds, dict_character)
    conf = float(np.max(preds, axis=1).mean()) if preds.shape[0] else 0.0
    return text, conf


def ocr_image(pipe_args, img_path, device, max_w=None, dyn_cap=960):
    """Full pipeline over one image. Returns list of (box, text, conf)."""
    core, det_c, rec_c, dict_character = pipe_args
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")

    det_out = det_c.output(0)
    rec_out = rec_c.output(0)

    t0 = time.perf_counter()
    boxes = detect_text_regions(det_c, det_out, img)
    t_det = time.perf_counter()

    results = []
    for (x, y, w, h) in boxes:
        crop = img[y:y + h, x:x + w]
        text, conf = recognize_text(rec_c, rec_out, crop, dict_character,
                                    max_w=max_w, dyn_cap=dyn_cap)
        results.append(((x, y, w, h), text, conf))
    t_rec = time.perf_counter()

    return results, t_det - t0, t_rec - t_det


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="PP-OCRv4 OCR over OpenVINO")
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="CPU",
                    choices=["CPU", "NPU", "GPU", "AUTO", "CPU", "GPU.0"])
    ap.add_argument("--dir", default="models/ppocrv4")
    ap.add_argument("--det-width", type=int, default=960)
    ap.add_argument("--det-height", type=int, default=960)
    ap.add_argument("--rec-width", type=int, default=None,
                    help="Fixed rec width for NPU; leave unset for CPU/GPU")
    args = ap.parse_args()

    device = args.device.upper()
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent.parent / model_dir

    print(f"[load] device={device} det={args.det_width}x{args.det_height} "
          f"rec_width={args.rec_width}", flush=True)
    t0 = time.perf_counter()
    pipe = load_models(model_dir, device,
                       det_shape=(args.det_height, args.det_width),
                       rec_fixed_width=args.rec_width)
    print(f"[load] compiled in {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    results, t_det, t_rec = ocr_image(pipe, args.image, device,
                                      max_w=args.rec_width, dyn_cap=960)
    total = time.perf_counter() - t0

    print(f"\n=== OCR result: {args.image} (device={device}) ===")
    print(f"detected {len(results)} text region(s)")
    for (x, y, w, h), text, conf in results:
        print(f"  [{x},{y} {w}x{h}] conf={conf:.3f} : {text!r}")
    print(f"\n[timer] total={total*1000:.1f}ms  det={(t_det)*1000:.1f}ms  "
          f"rec={(t_rec)*1000:.1f}ms")
    return results


if __name__ == "__main__":
    main()
