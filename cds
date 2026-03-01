#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"
PYTHON_BIN="${CDS_PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
catDetectionSystem command interface

Usage:
  cds detect --uri <uri-or-path> [options]
  cds infer --config <runtime-config> [options]
  cds train [--config <train-config>] [train options]
  cds evaluate --config <eval-config> [options]
  cds export --config <export-config> [options]
  cds dataset <prepare|validate> [options]
  cds monitor [options]
  cds doctor --target <runtime|training|all> [options]
  cds detect-c4
  cds help

Commands:
  detect      Run modular inference runtime (Stage 1).
  infer       Alias for detect using runtime config flows.
  train       Stage 2 training and active-learning commands.
  evaluate    Stage 2 evaluation and production gating.
  export      Stage 2 model export orchestration.
  dataset     Stage 2 dataset conversion and validation.
  monitor     Show runtime environment telemetry.
  doctor      Probe runtime or training toolchain capabilities.
  detect-c4   Legacy C4 RTSP wrapper (deprecated).
  help        Show this help text.
EOF
}

command_name="${1:-help}"

case "${command_name}" in
    detect|infer|train|evaluate|export|dataset|monitor|doctor)
        exec "${PYTHON_BIN}" -m cds "$@"
        ;;
    detect-c4)
        shift
        echo "[DEPRECATED] detect-c4 is a compatibility wrapper."
        echo "[DEPRECATED] Prefer: cds detect --uri <rtsp-uri>"
        exec "${SCRIPT_DIR}/run-c4.sh" "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        usage
        exit 1
        ;;
esac
