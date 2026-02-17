#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="artifacts/models/communitycats-prod-20260217-201910/checkpoints/best.mlpackage"
#MODEL="yolov8s.pt"

for video in /Volumes/camera/communitycats/referenceVideos/*.mp4; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --rate-limit-fps 120 \
    --queue-size 2 --imgsz 416 --confidence 0.8 --nms 0.75
done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
