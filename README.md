# o4-ocr

Local OCR for the Intel NPU.

I run Omarchy (Arch + Hyprland), and its built-in text capture — press Super+Shift+T, drag a region, text lands on the clipboard — is a `tesseract` job on the CPU. Fast enough, but every time I use it I look at the laptop's idle NPU and it feels like a waste. Intel's AI Boost unit is a ~40 TOPS accelerator sitting beside a Core Ultra 5 226V, and screenshot OCR is basically the textbook workload for it.

That premise became this little project: replace tesseract with a real on-device OCR model and find out whether the NPU earns its keep, or whether "NPU support" on a model card is just a checkbox.

## The model

`Intel/ocr-text-recognition` on Hugging Face — a PP-OCRv4 pipeline (DBNet text detection + CRNN-CTC recognition) converted to OpenVINO IR, listed as running on CPU/GPU/NPU.

One thing caught me early: the HF repo doesn't actually ship the weights. It has a build script, a README, and two 29 MB demo GIFs. The real models come from PaddlePaddle's servers, so "download the model" really means:

1. pull the `ch_PP-OCRv4_det_infer` and `ch_PP-OCRv4_rec_server_infer` tarballs — I used the *server* recognizer, since the model card says it's measurably better on decorative/photo text than the mobile variant,
2. convert both to FP16 OpenVINO IR with `ovc`,
3. grab the `ppocr_keys_v1.txt` character dictionary for CTC decoding.

All of that lives in `scripts/fetch_models.sh`. I keep the weight binaries out of git — they're fully reproducible, and nobody wants to clone an 86 MB file just to read a README.

## Hardware

| component | spec |
|---|---|
| CPU | Intel Core Ultra 5 226V |
| GPU | Intel Arc 130V (integrated) |
| NPU | Intel AI Boost (~40 TOPS) |

`ov::Core().available_devices` → `['CPU', 'GPU', 'NPU']` · NPU on `/dev/accel/accel0`.

## Layout

```
data/                        test images (synthetic screenshots + a real photo) + ground truth
models/ppocrv4/              converted IR graphs (.xml, small) + PP-OCRv4 char dict
src/ocr_pipeline.py          full det + rec pipeline over OpenVINO; --device CPU|NPU|GPU|AUTO
src/bench.py                 CPU vs NPU latency + character-accuracy table
src/gen_data.py              regenerate the synthetic screenshot samples
scripts/fetch_models.sh      download PaddlePaddle weights -> convert to FP16 IR
scripts/omarchy_ocr_npu.sh   drop-in NPU replacement for `omarchy capture text`
```

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install openvino opencv-python-headless pillow
./scripts/fetch_models.sh

python src/ocr_pipeline.py --image data/screenshot_terminal.png --device NPU --rec-width 1024
python src/bench.py --devices CPU,NPU
```

Gotcha worth writing down: the recognizer's input has a dynamic width axis, and the NPU only accepts static shapes. So on NPU every crop gets padded to a fixed `--rec-width`. I landed on 1024 — long screenshot lines were getting their tails chopped at the default that PaddleOCR tends to lean toward.

## What I measured

| image | device | det (ms) | rec (ms) | total (ms) | char-acc |
|---|---|---|---|---|---|
| terminal shot | CPU | 42 | 6466 | **6507** | 100% |
| terminal shot | **NPU** | 22 | 71 | **93** | 100% |
| code shot | CPU | 49 | 6430 | **6480** | 96.3% |
| code shot | **NPU** | 61 | 63 | **123** | 96.3% |
| UI shot | CPU | 41 | 4467 | **4508** | 96.8% |
| UI shot | **NPU** | 24 | 41 | **65** | 95.7% |
| real photo (11 lines) | CPU | 42 | 19605 | **19647** | — |
| real photo (11 lines) | **NPU** | 25 | 135 | **160** | — |

And the incumbent — Omarchy's stock text capture (`tesseract`, CPU):

| image | total | char-acc |
|---|---|---|
| terminal shot | 377 ms | 100% |
| code shot | 325 ms | 97.1% |
| UI shot | 268 ms | 100% |
| real photo | 768 ms | — |

Method notes, so the numbers aren't hand-wavy: accuracy is `1 − (Levenshtein distance / max length)` against a `ground_truth.json` I wrote, over synthetic screenshots I render from DejaVu Sans plus one real-world photo (`test_ocr.jpg`). Times are wall-clock for a full det+rec pass on the hardware listed above.

## Takeaways

- **The recognition stage is ~100× faster on the NPU than the CPU** — 4.4–19.6 *seconds* of CPU inference drops to 41–135 *milliseconds*. A screen grab goes from "wait for it" to "it's already on the clipboard."
- Accuracy is a wash on clean text (both engines sit at 96–100% against my ground truth), so the NPU costs nothing in read quality.
- `tesseract` stays a fine zero-dependency fallback. The real argument for the NPU path isn't accuracy — it's that a hotkey used constantly should feel instant, keep the CPU free for the compositor, and stay 100% local/offline.

## Not done yet

- Actually swapping `scripts/omarchy_ocr_npu.sh` in as the real capture command.
- Trying the *mobile* recognizer (~10 MB vs ~86 MB) to see what accuracy actually costs.
- INT8 quantization via NNCF and a second benchmark pass — the NPU tends to warm to that even more.

## License

MIT — matches Intel's PP-OCRv4 licensing.

---

*Hobby project, built with Claude Code (running Claude Fable 5). The measurements above are real runs on the hardware listed.*
