#!/usr/bin/env bash

MODEL="YOLOv3.mlmodel"
MODEL="yolov5xu.pt"
MODEL="artifacts/models/communitycats-prod-20260225-065559/ultralytics_train/weights/last.pt"
video="/Users/dpd/Movies/tapo-cat1-2026.02.24/all.mp4"
video="/Users/dpd/Movies/tapo-cat1-2026.02.23/all.mp4"

./cds detect --uri $video \
    --model-path $MODEL \
    --imgsz 320 --nms 0.5 \
    --confidence 0.6 \
    --benchmark 
#     --confidence-min 0.60 \
#     --export-frames \
#     --export-frames-dir artifacts/exports/active-learning \
#     --export-frames-sample-pct 25 \
