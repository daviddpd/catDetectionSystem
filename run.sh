#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] run.sh is a legacy wrapper."
echo "[DEPRECATED] Use: ${SCRIPT_DIR}/cds detect --uri <uri-or-path> [other options]"

for video in /Volumes/camera/uploads/c4/2026/02/16/; do
    echo "$video"
    python3 "${SCRIPT_DIR}/rtsp-object-ident.py" \
        --repoPath "${SCRIPT_DIR}" \
        --communityCatsPath /Volumes/camera/communitycats \
        --uri "$video"
done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
