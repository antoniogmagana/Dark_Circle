#!/bin/bash

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
echo "  5) InceptionTime"
echo "  6) BiGRU"
echo "  7) ALL MODELS"
echo ""
read -p "Models > " model_input

selected_models=()
if [[ "$model_input" == *"7"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "InceptionTime" "BiGRU")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("InceptionTime")
    [[ "$model_input" == *"6"* ]] && selected_models+=("BiGRU")
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

# --- 4. Sensor Selection ---
echo ""
echo "============================================================"
echo " SELECT SENSORS"
echo "============================================================"
echo "Enter the numbers of the sensors to train (space-separated)."
echo "Or type 'all' to select everything."
echo ""
echo "  1) audio"
echo "  2) seismic"
echo "  3) ALL SENSORS"
echo ""
read -p "Sensors > " sensor_input

selected_sensors=()
if [[ "$sensor_input" == *"3"* ]] || [[ "${sensor_input,,}" == *"all"* ]]; then
    selected_sensors=("audio" "seismic")
else
    [[ "$sensor_input" == *"1"* ]] && selected_sensors+=("audio")
    [[ "$sensor_input" == *"2"* ]] && selected_sensors+=("seismic")
fi

if [ ${#selected_sensors[@]} -eq 0 ]; then
    echo ""
    echo "[!] Error: You must select at least one sensor."
    exit 1
fi

echo ""
echo "Sensors to train: ${selected_sensors[*]}"

# --- 5. Unified Model Training Pipeline (parallel by sensor within each mode) ---

# Compute per-job worker budget (36 workers total, divided equally across concurrent jobs)
_n_jobs=$(( ${#selected_models[@]} * ${#selected_sensors[@]} ))
_workers_per_job=$(( 36 / _n_jobs ))
if (( _workers_per_job < 1 )); then _workers_per_job=1; fi
echo ""
echo "Worker budget: 36 total / $_n_jobs jobs = $_workers_per_job workers/job"

echo ""
for mode in "${selected_modes[@]}"; do
    echo "============================================================"
    echo "MODE: $mode — launching sensor jobs in parallel"
    echo "============================================================"

    # Launch all seismic jobs in the background
    if [[ " ${selected_sensors[*]} " =~ " seismic " ]]; then
        for model in "${selected_models[@]}"; do
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
            echo "  [BG] seismic | $model | $CURRENT_RUN_ID"
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode TRAIN_SENSOR=seismic MODEL_NAME=$model \
                NUM_WORKERS=$_workers_per_job \
                poetry run python train.py &
            sleep 1
        done
    fi

    # Launch all audio jobs in the background (concurrent with seismic)
    if [[ " ${selected_sensors[*]} " =~ " audio " ]]; then
        for model in "${selected_models[@]}"; do
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)
            echo "  [BG] audio   | $model | $CURRENT_RUN_ID"
            RUN_ID=$CURRENT_RUN_ID TRAINING_MODE=$mode TRAIN_SENSOR=audio MODEL_NAME=$model \
                NUM_WORKERS=$_workers_per_job \
                poetry run python train.py &
            sleep 1
        done
    fi

    echo "  Waiting for all $mode jobs to finish..."
    wait
    echo "  $mode complete."
done

# --- 6. Evaluation ---
echo "============================================================"
echo "ALL TRAINING COMPLETE. COMMENCING BATCH EVALUATION..."
echo "============================================================"

# Per-sensor eval, fused eval, then aggregate
poetry run python eval.py
poetry run python aggregate_results.py

echo "============================================================"
echo "PIPELINE FULLY COMPLETE."
echo "============================================================"