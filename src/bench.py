#!/usr/bin/env python3
"""
Benchmark PP-OCRv4 OCR (Intel/ocr-text-recognition) across OpenVINO devices.

Runs each test image through CPU and NPU, reports per-image latency and
char-level accuracy versus ground_truth.json, and prints a comparison table.

Usage:
    python bench.py [--devices CPU,NPU] [--rec-width NPU] [--dir models/ppocrv4]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import ocr_pipeline as op


def accuracy(pred_lines, gt):
    """Char-level accuracy (1 - Levenshtein distance / max(len))."""
    def lev(a, b):
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                               prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    pred = " ".join(pred_lines)
    gt_flat = " ".join(gt.split())
    if not gt_flat:
        return 0.0
    dist = lev(pred, gt_flat)
    return max(0.0, 1.0 - dist / max(len(pred), len(gt_flat)))


def run_once(pipe, image, device, rec_width):
    img = __import__("cv2").imread(str(image))
    t0 = time.perf_counter()
    boxes = op.detect_text_regions(pipe[1], pipe[1].output(0), img,
                                   det_shape=(960, 960))
    t1 = time.perf_counter()
    texts = []
    t2 = t1
    for (x, y, w, h) in boxes:
        crop = img[y:y + h, x:x + w]
        text, _ = op.recognize_text(pipe[2], pipe[2].output(0), crop,
                                    pipe[3], max_w=rec_width, dyn_cap=960)
        texts.append(text)
    t2 = time.perf_counter()
    return texts, (t1 - t0), (t2 - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="CPU,NPU")
    ap.add_argument("--rec-width", type=int, default=None,
                    help="Overrides fixed rec width for every device")
    ap.add_argument("--rec-width-cpu", type=int, default=None,
                    help="Fixed rec width for CPU (None=dynamic)")
    ap.add_argument("--rec-width-npu", type=int, default=1024,
                    help="Fixed rec width for NPU")
    ap.add_argument("--dir", default="models/ppocrv4")
    args = ap.parse_args()

    def rec_width_for(dev):
        if args.rec_width is not None:
            return args.rec_width
        return args.rec_width_npu if dev == "NPU" else args.rec_width_cpu

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    model_dir = Path(args.dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    data_dir = ROOT / "data"

    gt = json.load(open(data_dir / "ground_truth.json", encoding="utf-8"))
    images = sorted(p for p in (data_dir / "test_ocr.jpg",
                                *data_dir.glob("screenshot_*.png")))
    devices = [d.strip().upper() for d in args.devices.split(",")]

    # Pre-compile each device pipeline once.
    pipes = {}
    for dev in devices:
        print(f"[compile] {dev}", flush=True)
        pipes[dev] = op.load_models(model_dir, dev,
                                    det_shape=(960, 960),
                                    rec_fixed_width=rec_width_for(dev))

    rows = []
    for img in images:
        base = img.name
        gt_txt = gt.get(base, "")
        for dev in devices:
            texts, t_det, t_rec = run_once(pipes[dev], img, dev,
                                           rec_width_for(dev))
            acc = accuracy(texts, gt_txt) if gt_txt else float("nan")
            rows.append((base, dev, t_det * 1000, t_rec * 1000,
                         (t_det + t_rec) * 1000, acc))

    print("\n{:<24} {:<5} {:>9} {:>9} {:>9} {:>9}".format(
        "image", "dev", "det(ms)", "rec(ms)", "total(ms)", "acc"))
    print("-" * 66)
    for base, dev, d, r, tot, acc in rows:
        a = f"{acc * 100:.1f}%" if acc == acc else "  -"
        print("{:<24} {:<5} {:>8.1f} {:>8.1f} {:>8.1f} {:>9}".format(
            base, dev, d, r, tot, a))

    print("\n(CPU rec width = "
          f"{rec_width_for('CPU') or 'dynamic'}, "
          f"NPU rec width = {rec_width_for('NPU')})")


if __name__ == "__main__":
    main()
