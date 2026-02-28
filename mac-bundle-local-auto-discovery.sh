python3 /Users/dpd/Documents/projects/github/catDetectionSystem/artifacts/models/communitycats-prod-20260217-213759/rknn/make_calibration_txt.py \
  /Volumes/camera/communitycats/custom_data/imagebyclass/ \
  --output /Users/dpd/Documents/projects/github/catDetectionSystem/artifacts/models/communitycats-prod-20260217-213759/rknn/calibration.txt \
  --use-bundle-model \
  --labels-path /Users/dpd/Documents/projects/github/catDetectionSystem/config/classes-communitycats-prod-20260217-213759.txt \
  --backend auto \
  --imgsz 416 \
  --min-confidence 0.90 \
  --coverage-per-label 1 \
  --limit 1000

