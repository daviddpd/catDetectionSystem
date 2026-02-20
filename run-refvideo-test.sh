#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="artifacts/models/communitycats-prod-20260217-213759/exports/best.mlpackage"
#MODEL="yolov8s.pt"

for video in /Volumes/camera/communitycats/referenceVideos/*.mp4; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz 416 --confidence 0.5 --nms 0.6 \
    --log-level DEBUG  --benchmark 
done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
