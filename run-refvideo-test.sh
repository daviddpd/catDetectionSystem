#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS=`uname -o`

# Working Model artifacts directory
WMD="artifacts/models/x-community-cats-20260309-065444"

if [ $OS == "Darwin" ]; then
    MODEL="$WMD/exports/best.mlpackage"
else
    MODEL="$WMD/rknn/model.toolkit2.rknn"
fi
echo " ================ Select Model ==========================="
echo "Model: $MODEL"
echo " ========================================================="





#video_dir="/Volumes/camera/communitycats/referenceVideos/*.mp4"
#video_dir=`ls -1 /Volumes/camera/communitycats/referenceVideos*/*.mp4 | sort -R | xargs`
#video_dir=`ls -1 /Volumes/camera/communitycats/referenceVideos*/*.mp4 | xargs`
#video_dir="/Users/dpd/Movies/tapo-cat1-2026-03-02/all.mp4"
# video_dir=`find /Volumes/camera/C3 -name "*.mkv" | sort -R | xargs`
#video_dir="/Users/dpd/Movies/tapo-cat1-2026-03-02/7820519622EE_20260302144356809_cam0.mp4"
#video_dir="/Users/dpd/Movies/tapo-cat1/*.mp4"
#video_dir="/Users/dpd/Movies/tapo-cat1-2026.02.24/all.mp4"
#video_dir="/Users/dpd/Movies/tapo-cat1-2026-03-08/all.mp4"
#video_dir="/Users/dpd/Olives-prey-the-feathertoy.mov"
# 0004-C3_01_20211002230814: Opossum false positive as a cat. 
#video_dir="/Volumes/camera/communitycats/referenceVideos/0004-C3_01_20211002230814.mp4"



if [ $OS == "Darwin" ]; then
    video_dir="/Users/dpd/Movies/cds-demo-video.mp4"
else
    video_dir="/z/camera/communitycats/cds-demo-video.mp4"
fi

for video in $video_dir; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz 320 --nms 0.5 \
    --confidence 0.5 \
    --benchmark --no-event-stdout
#         --confidence-min 0.50 \
#         --export-frames \
#         --export-frames-dir artifacts/exports/active-learning-2026.02.24 \
#         --export-frames-sample-pct 25 \

done

# Example:
# ${SCRIPT_DIR}/cds detect --communityCatsPath /Volumes/camera/communitycats --uri rtsp://<camera-uri>
