#!/bin/sh

DATASET="/Users/dpd/Documents/projects/communitycats/imagebyclass"
OUTPUTROOT="/Users/dpd/Documents/projects/communitycats/imagebyclass-2026-03-08"

./cds dataset prepare \
  --xml-root $DATASET \
  --image-root $DATASET \
  --output-root $OUTPUTROOT \
  --config $DATASET/config/dataset.yaml \
  --split-mode deterministic
./cds dataset validate --dataset-root $OUTPUTROOT
./cds train --config $DATASET/config/train.yaml \
            --dataset $OUTPUTROOT/data.yaml \
            --from-scratch --model-arch config/model_arch/yolov8s.yaml \
            --device mps \
            --export-targets all
            

