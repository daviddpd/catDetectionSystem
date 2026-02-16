#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
catDetectionSystem command wrapper

Usage:
  cds detect --uri <uri-or-path> [options]
  cds detect-c4
  cds help

Commands:
  detect      Run the legacy RTSP/file detector entrypoint.
  detect-c4   Run the legacy C4 RTSP example wrapper.
  help        Show this help text.
EOF
}

command_name="${1:-help}"

case "${command_name}" in
    detect)
        shift
        exec python3 "${SCRIPT_DIR}/rtsp-object-ident.py" --repoPath "${SCRIPT_DIR}" "$@"
        ;;
    detect-c4)
        shift
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
