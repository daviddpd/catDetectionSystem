#!/usr/bin/env bash
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECT_SRC=$1

# Working Model artifacts directory
WMD="artifacts/models/x-community-cats-20260309-065444"
WMD="artifacts/models/x-community-cats-20260313-002814"
WMD="artifacts/models/x-community-cats-20260316-084322"

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
    CONFIDENCE=0.52
    MOUNTPT="/Volumes"
else
    MODEL="$WMD/rknn/model.toolkit2.rknn"
    if [ ! -f "$MODEL" ]; then
	    MODEL="$WMD/rknn/model.toolkit2.vendor.rknn"
	    if [ ! -f "$MODEL" ]; then
		  echo
		  echo " Failed to find the rknn model"
		  echo
	    fi
    fi
    IMGSIZE=320 
    NMS=0.3
    CONFIDENCE=0.72
    MOUNTPT="/z"
fi


demo_video_path="/Users/dpd/Movies/cds-demo-video.mp4"
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
                files=`find $2 -name '*.m[pk][4v]' |  xargs`
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
                files=`find $2 -name '*.m[pk][4v]' |  xargs`
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

# Host to Run headless with remote mjpeg
# --remote-mjpeg \
# --remote-host 0.0.0.0 \
#  --remote-port 8081 \
#  --remote-path /cats.mjpg
#
#  http://127.0.0.1:8080/stream.mjpg
#  http://<orange-pi-ip>:8080/stream.mjpg
#


case $2 in
        headless)
                OPTS="$OPTS  --remote-mjpeg --remote-host 0.0.0.0 --remote-port 8080 --remote-path /c100.mjpg"
        ;;
esac


echo " ================ Src ================================== "
echo "URIs: $src "
echo " ========================================================="

for video in $src; do
    echo "$video"
    ./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz $IMGSIZE --nms $NMS \
    --confidence $CONFIDENCE $BENCHMARK $NO_EVENT $OPTS
#         --confidence-min 0.50 \
#         --export-frames \
#         --export-frames-dir artifacts/exports/active-learning-2026.02.24 \
#         --export-frames-sample-pct 25 \
done


