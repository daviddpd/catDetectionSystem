#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: tools/rknn_gst_smoke.sh --uri <path-or-rtsp> [--codec h264|h265] [--source-kind file|rtsp] [--print-only]

Builds a recommended Rockchip MPP GStreamer pipeline and either prints the gst-launch
command or executes it with fakesink for a decode smoke test.
EOF
}

URI=""
CODEC=""
SOURCE_KIND="auto"
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uri)
      URI="${2:-}"
      shift 2
      ;;
    --codec)
      CODEC="${2:-}"
      shift 2
      ;;
    --source-kind)
      SOURCE_KIND="${2:-}"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$URI" ]]; then
  usage >&2
  exit 2
fi

BUILD_ARGS=(python3 "$SCRIPT_DIR/rknn_build_gst_pipeline.py" --uri "$URI" --source-kind "$SOURCE_KIND" --format all)
if [[ -n "$CODEC" ]]; then
  BUILD_ARGS+=(--codec "$CODEC")
fi
GST_LINE=""
while IFS= read -r line; do
  if [[ "$line" == gst_launch=* ]]; then
    GST_LINE="${line#gst_launch=}"
  fi
done < <("${BUILD_ARGS[@]}")

if [[ -z "$GST_LINE" ]]; then
  echo "Failed to build gst-launch command." >&2
  exit 1
fi

echo "$GST_LINE"
if [[ "$PRINT_ONLY" -eq 1 ]]; then
  exit 0
fi

eval "$GST_LINE"
