#!/bin/bash

# Exit immediately if a pipeline fails entirely (optional but safe)
set -e 

for mode in detection category instance; do
    # --- Unified Model Training Pipeline ---
    for model in DetectionCNN ClassificationCNN WaveformClassificationCNN ClassificationLSTM IterativeMiniRocket; do
        CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
        
        echo "============================================================"
        echo "STARTING RUN -> MODE: $mode | MODEL: $model | RUN_ID: $CURRENT_RUN_ID"
        echo "============================================================"
        
        RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python train.py
        
        # Check if training succeeded before evaluating
        if [ $? -eq 0 ]; then
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python eval.py
        else
            echo "Training failed for $model in $mode mode. Skipping eval."
        fi
        
        # Force a 1-second delay to guarantee unique RUN_IDs if a run fails instantly
        sleep 1 
    done
done

echo "============================================================"
echo "ALL PIPELINES COMPLETE."
echo "============================================================"