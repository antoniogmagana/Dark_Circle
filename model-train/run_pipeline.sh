#!/bin/bash

# Exit immediately if a pipeline fails entirely
set -e 

for mode in detection category instance; do
    # --- Unified Model Training Pipeline ---
    for model in DetectionCNN ClassificationCNN WaveformClassificationCNN ClassificationLSTM IterativeMiniRocket; do
        CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
        
        echo "============================================================"
        echo "STARTING RUN -> MODE: $mode | MODEL: $model | RUN_ID: $CURRENT_RUN_ID"
        echo "============================================================"
        
        # We only need to run train.py in the loop now
        RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python train.py
        
        # Force a 1-second delay to guarantee unique RUN_IDs if a run fails instantly
        sleep 1 
    done
done

echo "============================================================"
echo "ALL TRAINING COMPLETE. COMMENCING BATCH EVALUATION..."
echo "============================================================"

# Run eval.py and aggregate_results.py once at the very end. 
# It will automatically find and process all 15 models we just trained.
poetry run python eval.py
poetry run python aggregate_results.py

echo "============================================================"
echo "PIPELINE FULLY COMPLETE."
echo "============================================================"