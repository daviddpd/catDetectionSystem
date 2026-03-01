#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  tools/rknn_benchmark_matrix.sh \
    --uri <video-or-rtsp> \
    [--model-640 <path>] \
    [--model-320 <path>] \
    [--labels-path <path>] \
    [--codec h264|h265] \
    [--gstreamer-pipeline "<pipeline>"] \
    [--confidence <float>] \
    [--nms <float>] \
    [--output-dir <dir>] \
    [--dry-run]

Runs the recommended benchmark matrix:
  1. pyav + display
  2. pyav + headless
  3. gstreamer + display
  4. gstreamer + headless
and repeats it for whichever model(s) are provided.

Each case writes a full log file plus a compact summary of:
  - fps_decode
  - fps_infer
  - frame_age_ms
  - configured_imgsz
  - effective model input
EOF
}

URI=""
MODEL_640=""
MODEL_320=""
LABELS_PATH=""
CODEC="h264"
GSTREAMER_PIPELINE=""
CONFIDENCE="0.6"
NMS="0.5"
OUTPUT_DIR=""
DRY_RUN=0
GSTREAMER_AVAILABLE=1
GSTREAMER_SKIP_REASON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uri)
      URI="${2:-}"
      shift 2
      ;;
    --model-640)
      MODEL_640="${2:-}"
      shift 2
      ;;
    --model-320)
      MODEL_320="${2:-}"
      shift 2
      ;;
    --labels-path)
      LABELS_PATH="${2:-}"
      shift 2
      ;;
    --codec)
      CODEC="${2:-}"
      shift 2
      ;;
    --gstreamer-pipeline)
      GSTREAMER_PIPELINE="${2:-}"
      shift 2
      ;;
    --confidence)
      CONFIDENCE="${2:-}"
      shift 2
      ;;
    --nms)
      NMS="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -z "$URI" || ( -z "$MODEL_640" && -z "$MODEL_320" ) ]]; then
  usage >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT_DIR/artifacts/benchmarks/rknn-rockchip-$STAMP"
fi
mkdir -p "$OUTPUT_DIR"

if [[ -z "$GSTREAMER_PIPELINE" ]]; then
  while IFS= read -r line; do
    if [[ "$line" == cds_pipeline=* ]]; then
      GSTREAMER_PIPELINE="${line#cds_pipeline=}"
    fi
  done < <(
    python3 "$SCRIPT_DIR/rknn_build_gst_pipeline.py" \
      --uri "$URI" \
      --codec "$CODEC" \
      --format all
  )
fi

if [[ -z "$GSTREAMER_PIPELINE" ]]; then
  echo "Failed to derive a GStreamer pipeline." >&2
  exit 1
fi

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
COMMANDS_FILE="$OUTPUT_DIR/commands.txt"
: > "$SUMMARY_FILE"
: > "$COMMANDS_FILE"

echo "benchmark_output_dir=$OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
echo "uri=$URI" | tee -a "$SUMMARY_FILE"
echo "gstreamer_pipeline=$GSTREAMER_PIPELINE" | tee -a "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

check_gstreamer_pipeline() {
  if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
    GSTREAMER_AVAILABLE=0
    GSTREAMER_SKIP_REASON="gst-launch-1.0 not found"
    return
  fi
  if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
    GSTREAMER_AVAILABLE=0
    GSTREAMER_SKIP_REASON="gst-inspect-1.0 not found"
    return
  fi

  local missing=()
  while IFS= read -r segment; do
    segment="$(trim_spaces "$segment")"
    [[ -z "$segment" ]] && continue
    local element="${segment%% *}"
    [[ -z "$element" ]] && continue
    if [[ "$element" == *"/"* ]]; then
      continue
    fi
    if ! gst-inspect-1.0 "$element" >/dev/null 2>&1; then
      missing+=("$element")
    fi
  done < <(printf '%s\n' "$GSTREAMER_PIPELINE" | tr '!' '\n')

  if [[ ${#missing[@]} -gt 0 ]]; then
    GSTREAMER_AVAILABLE=0
    GSTREAMER_SKIP_REASON="missing GStreamer elements: ${missing[*]}"
  fi
}

check_gstreamer_pipeline
if [[ "$GSTREAMER_AVAILABLE" -eq 0 ]]; then
  echo "gstreamer_status=skipped reason=$GSTREAMER_SKIP_REASON" | tee -a "$SUMMARY_FILE"
  echo >> "$SUMMARY_FILE"
fi

run_case() {
  local case_name="$1"
  local model_path="$2"
  local configured_imgsz="$3"
  local ingest_backend="$4"
  local headless_flag="$5"

  local log_file="$OUTPUT_DIR/${case_name}.log"
  if [[ ! -f "$model_path" ]]; then
    printf 'case=%s\n' "$case_name" | tee -a "$SUMMARY_FILE"
    echo "log_file=$log_file" | tee -a "$SUMMARY_FILE"
    echo "result=skipped reason=model_not_found path=$model_path" | tee -a "$SUMMARY_FILE"
    echo >> "$SUMMARY_FILE"
    return
  fi
  local -a cmd=(
    ./cds detect
    --uri "$URI"
    --backend rknn
    --model-path "$model_path"
    --imgsz "$configured_imgsz"
    --confidence "$CONFIDENCE"
    --nms "$NMS"
    --benchmark
    --no-event-stdout
    --log-level INFO
  )

  if [[ -n "$LABELS_PATH" ]]; then
    cmd+=(--labels-path "$LABELS_PATH")
  fi
  if [[ "$ingest_backend" == "gstreamer" ]]; then
    if [[ "$GSTREAMER_AVAILABLE" -eq 0 ]]; then
      printf 'case=%s\n' "$case_name" | tee -a "$SUMMARY_FILE"
      echo "log_file=$log_file" | tee -a "$SUMMARY_FILE"
      echo "result=skipped reason=$GSTREAMER_SKIP_REASON" | tee -a "$SUMMARY_FILE"
      echo >> "$SUMMARY_FILE"
      return
    fi
    cmd+=(--ingest-backend gstreamer --gstreamer-pipeline "$GSTREAMER_PIPELINE")
  else
    cmd+=(--ingest-backend pyav)
  fi
  if [[ "$headless_flag" == "1" ]]; then
    cmd+=(--headless)
  fi

  printf 'case=%s\n' "$case_name" | tee -a "$SUMMARY_FILE"
  printf '%q ' "${cmd[@]}" | tee -a "$COMMANDS_FILE"
  printf '\n' | tee -a "$COMMANDS_FILE"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "log_file=$log_file" | tee -a "$SUMMARY_FILE"
    echo "result=dry-run" | tee -a "$SUMMARY_FILE"
    echo >> "$SUMMARY_FILE"
    return
  fi

  local exit_code=0
  if (
    cd "$ROOT_DIR"
    "${cmd[@]}"
  ) >"$log_file" 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi

  echo "log_file=$log_file" | tee -a "$SUMMARY_FILE"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "result=error exit_code=$exit_code" | tee -a "$SUMMARY_FILE"
  fi
  python3 "$SCRIPT_DIR/rknn_benchmark_report.py" "$log_file" | tee -a "$SUMMARY_FILE"
  echo >> "$SUMMARY_FILE"
}

run_matrix_for_model() {
  local label="$1"
  local model_path="$2"
  local configured_imgsz="$3"

  run_case "${label}-pyav-display" "$model_path" "$configured_imgsz" "pyav" "0"
  run_case "${label}-pyav-headless" "$model_path" "$configured_imgsz" "pyav" "1"
  run_case "${label}-gst-display" "$model_path" "$configured_imgsz" "gstreamer" "0"
  run_case "${label}-gst-headless" "$model_path" "$configured_imgsz" "gstreamer" "1"
}

if [[ -n "$MODEL_640" ]]; then
  run_matrix_for_model "model640" "$MODEL_640" "640"
fi
if [[ -n "$MODEL_320" ]]; then
  run_matrix_for_model "model320" "$MODEL_320" "320"
fi

echo "summary_file=$SUMMARY_FILE"
echo "commands_file=$COMMANDS_FILE"
