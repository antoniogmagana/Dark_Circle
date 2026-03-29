#!/bin/bash

# Exit immediately if a pipeline fails entirely
set -e

# --- 1. Interactive Mode Selection ---
echo "============================================================"
echo " SELECT TRAINING MODES"
echo "============================================================"
echo "Enter the numbers of the modes you want to train (space-separated)."
echo "Or type 'all' to select everything."
echo ""
echo "  1) detection"
echo "  2) category"
echo "  3) instance"
echo "  4) ALL MODES"
echo ""
read -p "Modes > " mode_input

selected_modes=()
if [[ "$mode_input" == *"4"* ]] || [[ "${mode_input,,}" == *"all"* ]]; then
    selected_modes=("detection" "category" "instance")
else
    [[ "$mode_input" == *"1"* ]] && selected_modes+=("detection")
    [[ "$mode_input" == *"2"* ]] && selected_modes+=("category")
    [[ "$mode_input" == *"3"* ]] && selected_modes+=("instance")
fi

# --- 2. Interactive Model Selection ---
echo ""
echo "============================================================"
echo " SELECT MODELS"
echo "============================================================"
echo "Enter the numbers of the models you want to train (space-separated)."
echo "Or type 'all' to select everything."
echo ""
echo "  1) DetectionCNN"
echo "  2) ClassificationCNN"
echo "  3) WaveformClassificationCNN"
echo "  4) ClassificationLSTM"
echo "  5) IterativeMiniRocket"
echo "  6) InceptionTime"
echo "  7) TCN"
echo "  8) BiGRU"
echo "  9) ALL MODELS"
echo ""
read -p "Models > " model_input

selected_models=()
if [[ "$model_input" == *"9"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "IterativeMiniRocket" "InceptionTime" "TCN" "BiGRU")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("IterativeMiniRocket")
    [[ "$model_input" == *"6"* ]] && selected_models+=("InceptionTime")
    [[ "$model_input" == *"7"* ]] && selected_models+=("TCN")
    [[ "$model_input" == *"8"* ]] && selected_models+=("BiGRU")
fi

# --- 3. Validation & Confirmation ---
if [ ${#selected_modes[@]} -eq 0 ] || [ ${#selected_models[@]} -eq 0 ]; then
    echo ""
    echo "[!] Error: You must select at least one valid mode and one valid model."
    exit 1
fi

echo ""
echo "============================================================"
echo " RUN SUMMARY"
echo "============================================================"
echo "Modes to train:  ${selected_modes[*]}"
echo "Models to train: ${selected_models[*]}"
echo ""
read -p "Proceed with this configuration? (y/n) > " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Training aborted by user."
    exit 0
fi

# --- 4. Unified Model Training Pipeline ---
echo ""
for mode in "${selected_modes[@]}"; do
    for model in "${selected_models[@]}"; do
        CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)

        echo "============================================================"
        echo "STARTING RUN -> MODE: $mode | MODEL: $model | RUN_ID: $CURRENT_RUN_ID"
        echo "============================================================"

        RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode MODEL_NAME=$model poetry run python train.py

        sleep 1
    done
done

# --- 5. Individual Model Evaluation ---
echo "============================================================"
echo "ALL TRAINING COMPLETE. COMMENCING BATCH EVALUATION..."
echo "============================================================"

poetry run python eval.py
poetry run python aggregate_results.py

# --- 6. Ensemble Evaluation ---
echo ""
echo "============================================================"
echo "RUNNING ENSEMBLE EVALUATION..."
echo "============================================================"

for mode in "${selected_modes[@]}"; do
    echo ""
    echo "--- Ensemble: $mode ---"
    TRAINING_MODE=$mode poetry run python ensemble.py "$mode"
done

echo ""
echo "============================================================"
echo "PIPELINE FULLY COMPLETE."
echo "============================================================"
