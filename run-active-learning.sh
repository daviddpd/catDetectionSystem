#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECT_SRC=$1

# Working Model artifacts directory
WMD="artifacts/models/x-community-cats-20260309-065444"
WMD="artifacts/models/x-community-cats-20260313-002814"

OS=`uname -o`
BENCHMARK=""

if [ -n "$_CDS_BENCHMARK" ]; then
    BENCHMARK="--benchmark"
fi
if [ -n "$_CDS_NO_EVENT" ]; then
    NO_EVENT="--no-event-stdout"
fi

if [ $OS == "Darwin" ]; then
    MODEL="$WMD/exports/best.mlpackage"
    IMGSIZE=320 
    NMS=0.5
    CONFIDENCE=0.25
    MOUNTPT="/Volumes"
else
    MODEL="$WMD/rknn/model.toolkit2.rknn"
    IMGSIZE=320 
    NMS=0.5
    CONFIDENCE=0.5
    MOUNTPT="/z"
fi


demo_video_path="/Users/dpd/Movies/cds-demo-video.mp4 /z/camera/communitycats/cds-demo-video.mp4"
video_ref_dir="$MOUNTPT/camera/communitycats/referenceVideos $MOUNTPT/camera/communitycats/referenceVideos2"

echo " ================ Model ================================== "
echo "Model: $MODEL"
echo " ========================================================="


src=""

case $SELECT_SRC in
    yolo3)
        MODEL="YOLOv3.mlmodel"
        IMGSIZE=416
        OPTS="--labels-path config/yolo3-classes.txt"
        if [ -n "$2" ]; then
            if [ -d "$2" ]; then
                files=`find $2 -name '*.m[pk][4v]' | sort | xargs`
                src="$src $files"
            elif [ -f "$2" ]; then
                src="$2"
            fi
        fi
        ;;
    [cC]1|tplink)
        src='rtsp://admin:cwvqYgGn4vjGN3oKYdVBj@c1.dpdtech.com:554/h264Preview_01_main'
        ;;
    c100)
        src='rtsp://camera:GKouGCXCXmR2HzF@C100.dpdtech.com:554/stream1'
        ;;
    d1)
        src='rtsp://thingino:thingino@ing-cinnado-d1-268c.local/ch0'
        ;;
    demo)
        for v in $demo_video_path; do
            if [ -f "$v" ]; then
                src=$v
            fi
        done        
        ;;
    ref)
        for v in $video_ref_dir; do
            if [ -d "$v" ]; then
                files=`find $v -name '*.m[pk][4v]' | sort -R | xargs`
                src="$src $files"
            fi
        done
        ;;
    dir)
        if [ -n "$2" ]; then
            if [ -d "$2" ]; then
                files=`find $2 -name '*.m[pk][4v]' | sort | xargs`
                src="$src $files"
            elif [ -f "$2" ]; then
                src="$2"
            fi
        fi
        ;;
    *)
        echo "Usage $1 <shortcut> (<file|dir>)" 
        exit 1;
    ;;
esac

echo " ================ Src ================================== "
echo "URIs: $src "
echo " ========================================================="

for video in $src; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL $OPTS \
    --imgsz $IMGSIZE --nms $NMS \
    --benchmark \
    --confidence 0.40 \
    --confidence-min 0.52 \
    --export-frames \
    --export-frames-dir artifacts/exports/active-learning-2026.03.15 \
    --export-frames-sample-pct 1
done


