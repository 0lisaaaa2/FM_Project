#!/bin/bash

DATASETS=("animals" "braintumor" "pneumonia" "baggage" "bones_fixed")

DATASET_ROOT="/home/luca/git/datasets"
CLASSIFIER_SCRIPT="/home/luca/git/FM_Project/classifier.py"
EPOCHS=100
BATCH_SIZE=16

/home/luca/git/.fm_venv/bin/activate

for ds in "${DATASETS[@]}"; do
    echo "Running classifier for $ds"
    python "$CLASSIFIER_SCRIPT" \
        --datasetdir "$DATASET_ROOT/$ds/" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE"
done