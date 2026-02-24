#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="artifacts/models/communitycats-prod-20260217-213759/exports/best.mlpackage"
#MODEL="yolov8s.pt"

video_dir="/Users/dpd/Movies/tapo-cat1/*.mp4"
video_dir="/Volumes/camera/communitycats/referenceVideos/*.mp4"

for video in  $video_dir; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz 416 --nms 0.6 \
    --confidence 0.40 
#    --benchmark \
#     --confidence-min 0.60 \
#     --export-frames \
#     --export-frames-dir artifacts/exports/active-learning \
#     --export-frames-sample-pct 10 \
#     --no-event-stdout

done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
