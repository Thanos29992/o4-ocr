#!/usr/bin/env bash
# omarchy:summary=Extract text from a screenshot region with NPU-accelerated PP-OCRv4
# omarchy:group=capture
# omarchy:examples=omarchy capture text --npu
#
# Drop-in replacement for the stock `omarchy capture text` OCR, but runs the
# PaddleOCR PP-OCRv4 pipeline (Intel/ocr-text-recognition) on the Intel NPU
# via OpenVINO instead of tesseract on the CPU.
#
# Same UX: SUPER+SHIFT+T -> region select -> text on the clipboard.
# Streams the grim region PNG to the Python pipeline through a pipe so
# nothing is ever written to disk or sent anywhere.
#
# Requires:  src/ocr_pipeline.py + converted IR (run ./scripts/fetch_models.sh once)
# Optional:  OMARCHY_OCR_DEVICE (default NPU), OMARCHY_OCR_REC_WIDTH (default 1024)

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${OMARCHY_OCR_DEVICE:-NPU}"
RECW="${OMARCHY_OCR_REC_WIDTH:-1024}"
PY="${HERE}/.venv/bin/python"
PIPE="${HERE}/src/ocr_pipeline.py"

# Keep hyprpicker alive until after grim captures (freeze overlay)
hyprpicker -r -z >/dev/null 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
sleep .1

SELECTION=$(slurp 2>/dev/null)
[[ -z $SELECTION ]] && exit 0

# grim -> pipe image (PNG, stdout) -> python reads from stdin, runs NPU OCR
TEXT=$(grim -g "$SELECTION" - | "$PY" "$PIPE" --device "$DEVICE" --rec-width "$RECW" --pipe-stdin 2>/dev/null | tail -n +2) || exit 1

[[ -z $TEXT ]] && exit 1
printf "%s" "$TEXT" | wl-copy
omarchy-notification-send -g 󰘐 "Copied NPU OCR text to clipboard"
