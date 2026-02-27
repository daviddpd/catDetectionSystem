#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#MODEL="artifacts/models/communitycats-prod-20260217-213759/exports/best.mlpackage"



MODEL="artifacts/models/communitycats-prod-20260225-065559/exports/best.mlpackage"
if [ ! -f $MODEL ]; then
    MODEL="artifacts/models/communitycats-prod-20260225-065559/ultralytics_train/weights/last.pt"
fi

echo "Model: $MODEL"
#MODEL="yolov8s.pt"

#video_dir="/Volumes/camera/communitycats/referenceVideos/*.mp4"
video_dir=`ls -1 /Volumes/camera/communitycats/referenceVideos*/*.mp4 | sort -R | xargs`
#video_dir="/Users/dpd/Movies/tapo-cat1/*.mp4"
#video_dir="/Users/dpd/Movies/tapo-cat1-2026.02.24/all.mp4"
#video_dir="/Users/dpd/Movies/tapo-cat1-2026.02.23/all.mp4"

for video in $video_dir; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz 320 --nms 0.5 \
    --confidence 0.6 \
    --benchmark  
#         --confidence-min 0.50 \
#         --export-frames \
#         --export-frames-dir artifacts/exports/active-learning-2026.02.24 \
#         --export-frames-sample-pct 25 \

done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
