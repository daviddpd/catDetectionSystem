#!/bin/bash 

RUN_DIR="artifacts/models/x-community-cats-20260313-002814"
RUN_DIR="artifacts/models/x-community-cats-20260316-084322"

#./cds evaluate --config config/eval.yaml --model "$RUN_DIR/ultralytics_train/weights/best.pt"

./cds export \
  --config config/export.yaml \
  --model "$RUN_DIR/checkpoints/best.pt" \
  --output-dir "$RUN_DIR" \
  --targets all

  #--model "$RUN_DIR/ultralytics_train/weights/best.pt" \

#$RUN_DIR/rknn/make_calibration_txt.py --limit 250 --imgsz 320 /z/camera/communitycats/custom_data/imagebyclass/ --model-path artifacts/models/x-community-cats-20260309-065444/rknn/model.toolkit2.rknn
#
#cp artifacts/models/x-community-cats-20260313-002814/rknn/calibration.txt.bak artifacts/models/x-community-cats-20260313-002814/rknn/calibration.txt

cp $RUN_DIR/rknn/calibration.txt.bak $RUN_DIR/rknn/calibration.txt

$RUN_DIR/rknn/convert_toolkit2.py
