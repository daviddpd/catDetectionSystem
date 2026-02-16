#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] run-c4.sh is a legacy wrapper."
echo "[DEPRECATED] Use: ${SCRIPT_DIR}/cds detect --uri <rtsp-uri> [other options]"

python3 "${SCRIPT_DIR}/rtsp-object-ident.py" \
    --repoPath "${SCRIPT_DIR}" \
    --communityCatsPath /Volumes/camera/communitycats \
    --uri 'rtsp://admin:cwvqYgGn4vjGN3oKYdVBj@c4.dpdtech.com:554/h264Preview_01_main'
