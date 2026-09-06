#!/usr/bin/env python3
"""Generate synthetic screenshot-style test images with known ground truth."""
import json
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

cands = [
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
]
font_path = next((p for p in cands if os.path.exists(p)), None)
assert font_path, "no font found"
f = ImageFont.truetype(font_path, 36)

samples = {
    "screenshot_terminal.png": [
        "shlok@acer ~ $ pacman -Syu",
        ":: Synchronizing package databases...",
        " core is up to date",
        " extra is up to date",
        ":: Starting full system upgrade...",
    ],
    "screenshot_code.png": [
        'def ocr_pipeline(image, device):',
        '    boxes = detect_text_regions(image)',
        '    return recognize_all(boxes)',
        'latency_ms = bench(device)  # NPU target',
    ],
    "screenshot_ui.png": [
        "Nepal Notes   Save | Share",
        "The quick brown fox jumps over the lazy dog",
        "8080  connected  uptime 2h",
    ],
}

os.makedirs(DATA, exist_ok=True)
for name, lines in samples.items():
    W, H = 900, 40 + 46 * len(lines)
    img = Image.new("RGB", (W, H), (24, 26, 30))
    d = ImageDraw.Draw(img)
    y = 18
    for ln in lines:
        d.text((24, y), ln, font=f, fill=(230, 232, 236))
        y += 46
    img.save(os.path.join(DATA, name))
    print("wrote", name, img.size)

gt = {
    "screenshot_terminal.png":
        "shlok@acer ~ $ pacman -Syu :: Synchronizing package databases... "
        "core is up to date extra is up to date :: Starting full system "
        "upgrade...",
    "screenshot_code.png":
        'def ocr_pipeline(image, device): boxes = detect_text_regions(image) '
        'return recognize_all(boxes) latency_ms = bench(device)  # NPU target',
    "screenshot_ui.png":
        "Nepal Notes Save | Share The quick brown fox jumps over the lazy dog "
        "8080 connected uptime 2h",
}
with open(os.path.join(DATA, "ground_truth.json"), "w", encoding="utf-8") as fh:
    json.dump(gt, fh, ensure_ascii=False, indent=2)
print("wrote ground_truth.json")
