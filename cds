#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

usage() {
    cat <<'EOF'
catDetectionSystem command interface

Usage:
  cds detect --uri <uri-or-path> [options]
  cds monitor [options]
  cds doctor [options]
  cds detect-c4
  cds help

Commands:
  detect      Run modular inference runtime.
  monitor     Show runtime environment telemetry.
  doctor      Probe backend/decoder/runtime capabilities.
  detect-c4   Legacy C4 RTSP wrapper (deprecated).
  help        Show this help text.
EOF
}

command_name="${1:-help}"

case "${command_name}" in
    detect)
        exec python3 -m cds "$@"
        ;;
    monitor)
        exec python3 -m cds "$@"
        ;;
    doctor)
        exec python3 -m cds "$@"
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
