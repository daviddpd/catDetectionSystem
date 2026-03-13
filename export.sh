#!/bin/bash 

RUN_DIR="artifacts/models/x-community-cats-20260313-002814"

./cds evaluate --config config/eval.yaml --model "$RUN_DIR/ultralytics_train/weights/best.pt"

./cds export \
  --config config/export.yaml \
  --model "$RUN_DIR/ultralytics_train/weights/best.pt" \
  --output-dir "$RUN_DIR" \
  --targets all
