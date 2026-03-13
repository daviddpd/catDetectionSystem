#!/bin/sh


DATASET="/Users/dpd/Documents/projects/communitycats/imagebyclass"
OUTPUTROOT="/Users/dpd/Documents/projects/communitycats/imagebyclass-2026-03-12"

# Working Model artifacts directory
WMD="artifacts/models/x-community-cats-20260309-065444"
OS=`uname -o`

if [ $OS == "Darwin" ]; then
    MODEL="$WMD/exports/best.mlpackage"
    MODEL="$WMD/exports/model.pt"
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


# ./cds dataset prepare \
#   --xml-root $DATASET \
#   --image-root $DATASET \
#   --output-root $OUTPUTROOT \
#   --config $DATASET/config/dataset.yaml \
#   --split-mode deterministic
# 
# ./cds dataset validate --dataset-root $OUTPUTROOT --classes config/classes.txt 
# 
./cds train --config $DATASET/config/train.yaml \
            --dataset $OUTPUTROOT/data.yaml \
            --model $MODEL \
            --device mps \
            --export-targets all
            
