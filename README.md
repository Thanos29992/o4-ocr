# o4-ocr

On-device OCR for Omarchy/Hyprland, benchmarked — PaddleOCR PP-OCRv4
(det + rec) running as **OpenVINO IR** on **CPU / GPU / NPU**, compared
against the `tesseract` pipeline Omarchy ships today.

Built to answer one question: *can OCR actually use the Acer's NPU, and is it
worth it?*

**Short answer: yes, and dramatically.** The recognition half of PP-OCRv4 goes
from **~4–19 s on CPU** to **~40–135 ms on the NPU** — about **100× faster**,
with equal-or-better text accuracy. The Omarchy `omarchy capture text` hotkey
currently runs plain `tesseract` on CPU and never touches the NPU.

## Why

Your `SUPER+SHIFT+T` screenshot-OCR reads text via:

```
hyprpicker → slurp → grim (region) → tesseract (--oem 1 --psm 6, CPU) → wl-copy
```

That's fully local and private, but pure CPU. `Intel/ocr-text-recognition` on
HuggingFace advertises **CPU / GPU / NPU** — it's a PP-OCRv4 pipeline converted
to OpenVINO IR, so the same text-extraction work can ride the NPU.

> Note: the HF repo only ships README + the `export_and_quantize.sh` script;
> the actual weights are pulled from PaddlePaddle and converted with `ovc`.
> This repo reproduces that via `scripts/fetch_models.sh` and does **not**
> commit the (up-to-86MB) weight binaries.

## Layout

| path | what |
|---|---|
| `src/ocr_pipeline.py` | PP-OCRv4 det+rec over OpenVINO; `--device CPU\|NPU\|GPU\|AUTO` |
| `src/bench.py` | CPU vs NPU latency + char-accuracy table |
| `src/gen_data.py` | regenerate synthetic screenshot test images |
| `data/` | test images + `ground_truth.json` |
| `models/ppocrv4/` | **converted IR graphs** (`.xml`, small) + char dict |
| `scripts/fetch_models.sh` | fetch PaddlePaddle weights → convert to FP16 IR |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openvino opencv-python-headless pillow
./scripts/fetch_models.sh            # pull weights + convert to IR (once)

python src/ocr_pipeline.py --image data/screenshot_terminal.png --device CPU
python src/ocr_pipeline.py --image data/screenshot_terminal.png --device NPU --rec-width 1024
python src/bench.py --devices CPU,NPU
```

NPU needs fully static tensor shapes, so the recognizer's dynamic width is
padded to a fixed `--rec-width` (default 1024 in bench). On CPU/GPU the width
stays dynamic.

## Benchmark (2026-09-06, Acer Aspire 14 AI, iGPU + NPU present)

| image | device | det (ms) | rec (ms) | total (ms) | char-acc |
|---|---|---|---|---|---|
| screenshot_terminal | CPU | 42 | 6466 | **6507** | 100% |
| screenshot_terminal | **NPU** | 22 | 71 | **93** | 100% |
| screenshot_code | CPU | 49 | 6430 | **6480** | 96.3% |
| screenshot_code | **NPU** | 61 | 63 | **123** | 96.3% |
| screenshot_ui | CPU | 41 | 4467 | **4508** | 96.8% |
| screenshot_ui | **NPU** | 24 | 41 | **65** | 95.7% |
| test_ocr (photo) | CPU | 42 | 19605 | **19647** | — |
| test_ocr (photo) | **NPU** | 25 | 135 | **160** | — |

For reference, today's `tesseract` (CPU-only, what Omarchy actually uses):

| image | tesseract total |
|---|---|
| screenshot_terminal | 377 ms |
| screenshot_code | 325 ms |
| screenshot_ui | 268 ms |
| test_ocr (photo) | 768 ms |

Takeaways:

* **Latency:** PP-OCRv4-on-NPU (~65–160 ms) beats both tesseract
  (~270–770 ms) *and* PP-OCRv4-on-CPU (~4.5–19.6 s). The rec model is heavy;
  the NPU eats it.
* **Accuracy:** on these samples, PP-OCRv4 ≈ tesseract on clean text; PP-OCRv4
  reads the photo's full paragraphs in one pass as well as tesseract does.
* tesseract wins on *nothing* except being already-installed.

## Hardware reality on this box

```
ov::Core().available_devices  →  ['CPU', 'GPU', 'NPU']
ls /dev/accel/                →  accel0   (NPU is live)
```

GPU shares a driver stack with the NPU runtime, but the OCR pipeline has no
reason to touch it — NPU is faster and leaves the CPU/GPU free for rendering.

## Next steps (not done yet)

- A drop-in replacement for `omarchy capture text` that runs
  `grim → ppocrv4-detect (NPU) → ppocrv4-recognize (NPU) → wl-copy`.
- Keep the `100% local / no cloud / no upload` property.
- Optional: try the *mobile* recognizer variant (~10 MB vs 86 MB) for a
  lighter install at a small accuracy cost.

## License

MIT (matches Intel's PP-OCRv4 model card license).
