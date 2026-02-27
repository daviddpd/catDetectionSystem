#!/bin/sh

DATASET="/Users/dpd/Documents/projects/communitycats/imagebyclass"
OUTPUTROOT="imagebyclass-2026-02-24"

./cds dataset prepare \
  --xml-root $DATASET \
  --image-root $DATASET \
  --output-root $OUTPUTROOT \
  --config $DATASET/config/dataset.yaml \
  --split-mode deterministic
./cds dataset validate --dataset-root $OUTPUTROOT --config $DATASET/config/dataset.yaml 
./cds train --config $DATASET/config/train.yaml --dataset imagebyclass-2026-02-24/data.yaml --device mps --export-targets all

