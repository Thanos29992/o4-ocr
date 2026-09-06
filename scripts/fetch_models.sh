#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Fetch the PP-OCRv4 (Intel/ocr-text-recognition) models and convert them to
# OpenVINO IR locally. The weight binaries are intentionally NOT committed to
# the o4-ocr repo — run this once to reproduce them.
#
# Based on Intel's Intel/ocr-text-recognition export_and_quantize.sh.
#
# Usage:  ./scripts/fetch_models.sh
# Output: models/ppocrv4/ch_PP-OCRv4_{det_infer,rec_server_infer}/inference.{xml,bin}
#         models/ppocrv4/ppocr_keys_v1.txt
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DEST="$ROOT/models/ppocrv4"
mkdir -p "$DEST"

DET_URL="https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar"
REC_URL="https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar"
DICT_URL="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"

require() { command -v "$1" >/dev/null 2>&1 || { echo "need: $1"; exit 1; }; }
# Prefer curl (present on this Arch box); fall back to wget.
DL=""
if command -v curl >/dev/null 2>&1; then DL="curl -fSL -o"; else require wget; DL="wget -q -O"; fi
require ovc; require python3

# ---- detection -----------------------------------------------------------
if [[ ! -f "$DEST/ch_PP-OCRv4_det_infer/inference.xml" ]]; then
  echo "== download + convert detection model =="
  ( cd "$DEST" && $DL det.tar "$DET_URL" && tar xf det.tar && rm -f det.tar )
  ovc "$DEST/ch_PP-OCRv4_det_infer/inference.pdmodel" \
      --output_model "$DEST/ch_PP-OCRv4_det_infer/inference.xml" --compress_to_fp16
fi

# ---- recognition (server variant, more accurate) -------------------------
if [[ ! -f "$DEST/ch_PP-OCRv4_rec_server_infer/inference.xml" ]]; then
  echo "== download + convert recognition model =="
  ( cd "$DEST" && $DL rec.tar "$REC_URL" && tar xf rec.tar && rm -f rec.tar )
  ovc "$DEST/ch_PP-OCRv4_rec_server_infer/inference.pdmodel" \
      --output_model "$DEST/ch_PP-OCRv4_rec_server_infer/inference.xml" --compress_to_fp16
fi

# ---- char dictionary ------------------------------------------------------
if [[ ! -f "$DEST/ppocr_keys_v1.txt" ]]; then
  echo "== download character dictionary =="
  $DL "$DEST/ppocr_keys_v1.txt" "$DICT_URL"
fi

echo "Done. IR models under: $DEST"
ls -la "$DEST"/ch_PP-OCRv4_*/inference.* "$DEST"/ppocr_keys_v1.txt
