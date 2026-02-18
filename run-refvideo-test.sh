#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="artifacts/models/communitycats-prod-20260217-213759/exports/best.mlpackage"
#MODEL="yolov8s.pt"

for video in /Volumes/camera/communitycats/referenceVideos2/*.mp4; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --rate-limit-fps 30 \
    --queue-size 1 --imgsz 416 --confidence 0.8 --nms 0.75 --no-event-stdout
done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
