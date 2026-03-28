#!/bin/bash
set -e

# --- 0. Mode: Sweep vs Full ---
echo "============================================================"
echo " PIPELINE MODE"
echo "============================================================"
echo "  1) Full training (30 epochs, early stopping)"
echo "  2) Quick sweep  (5 epochs, fast architecture comparison)"
echo ""
read -p "Mode > " pipeline_mode

SWEEP_FLAG=""
if [[ "$pipeline_mode" == "2" ]]; then
    SWEEP_FLAG="SWEEP=5"
    echo ""
    echo "  Sweep mode: 5 epochs per run for fast comparison."
    echo "  Re-run promising models in full mode afterwards."
fi

# --- 1. Sensor Selection ---
echo ""
echo "============================================================"
echo " SELECT SENSORS"
echo "============================================================"
echo "  1) audio"
echo "  2) seismic"
echo "  3) accel"
echo "  4) ALL SENSORS"
echo ""
read -p "Sensors (space-separated) > " sensor_input

selected_sensors=()
if [[ "$sensor_input" == *"4"* ]] || [[ "${sensor_input,,}" == *"all"* ]]; then
    selected_sensors=("audio" "seismic" "accel")
else
    [[ "$sensor_input" == *"1"* ]] && selected_sensors+=("audio")
    [[ "$sensor_input" == *"2"* ]] && selected_sensors+=("seismic")
    [[ "$sensor_input" == *"3"* ]] && selected_sensors+=("accel")
fi

# --- 2. Mode Selection ---
echo ""
echo "============================================================"
echo " SELECT TRAINING MODES"
echo "============================================================"
echo "  1) detection"
echo "  2) category"
echo "  3) instance"
echo "  4) ALL MODES"
echo ""
read -p "Modes (space-separated) > " mode_input

selected_modes=()
if [[ "$mode_input" == *"4"* ]] || [[ "${mode_input,,}" == *"all"* ]]; then
    selected_modes=("detection" "category" "instance")
else
    [[ "$mode_input" == *"1"* ]] && selected_modes+=("detection")
    [[ "$mode_input" == *"2"* ]] && selected_modes+=("category")
    [[ "$mode_input" == *"3"* ]] && selected_modes+=("instance")
fi

# --- 3. Model Selection ---
echo ""
echo "============================================================"
echo " SELECT MODELS"
echo "============================================================"
echo "  1) DetectionCNN               (2D spectrogram)"
echo "  2) ClassificationCNN          (2D spectrogram)"
echo "  3) WaveformClassificationCNN  (1D waveform)"
echo "  4) ClassificationLSTM         (1D waveform)"
echo "  5) BiGRU                      (1D waveform)"
echo "  6) TCN                        (1D waveform)"
echo "  7) InceptionTime              (1D waveform)"
echo "  8) IterativeMiniRocket        (1D waveform)"
echo "  9) ALL MODELS"
echo ""
read -p "Models (space-separated) > " model_input

selected_models=()
if [[ "$model_input" == *"9"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "BiGRU" "TCN" "InceptionTime" "IterativeMiniRocket")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("BiGRU")
    [[ "$model_input" == *"6"* ]] && selected_models+=("TCN")
    [[ "$model_input" == *"7"* ]] && selected_models+=("InceptionTime")
    [[ "$model_input" == *"8"* ]] && selected_models+=("IterativeMiniRocket")
fi

# --- 4. Validate ---
if [ ${#selected_sensors[@]} -eq 0 ] || [ ${#selected_modes[@]} -eq 0 ] || [ ${#selected_models[@]} -eq 0 ]; then
    echo "[!] Select at least one sensor, one mode, and one model."
    exit 1
fi

total_batches=$(( ${#selected_sensors[@]} * ${#selected_modes[@]} ))
total_runs=$(( total_batches * ${#selected_models[@]} ))

echo ""
echo "============================================================"
echo " RUN SUMMARY"
echo "============================================================"
if [[ -n "$SWEEP_FLAG" ]]; then
    echo "Pipeline: QUICK SWEEP (5 epochs)"
else
    echo "Pipeline: FULL TRAINING (30 epochs + early stopping)"
fi
echo "Sensors: ${selected_sensors[*]}"
echo "Modes:   ${selected_modes[*]}"
echo "Models:  ${selected_models[*]}"
echo ""
echo "Data loads:   ${total_batches} (one per sensor × mode)"
echo "Model trains: ${total_runs}"
echo ""
read -p "Proceed? (y/n) > " confirm
[[ "${confirm,,}" != "y" ]] && echo "Aborted." && exit 0

# --- 5. Train (sensor × mode → all models in one process) ---
echo ""
batch_count=0
for sensor in "${selected_sensors[@]}"; do
    for mode in "${selected_modes[@]}"; do
        batch_count=$((batch_count + 1))

        echo ""
        echo "============================================================"
        echo "BATCH ${batch_count}/${total_batches}"
        echo "SENSOR: $sensor | MODE: $mode"
        echo "MODELS: ${selected_models[*]}"
        echo "============================================================"

        env $SWEEP_FLAG \
        TRAIN_SENSOR=$sensor \
        TRAINING_MODE=$mode \
        MODEL_NAME=${selected_models[0]} \
            poetry run python train_batch.py "${selected_models[@]}"
    done
done

# --- 6. Evaluate all models ---
echo ""
echo "============================================================"
echo "TRAINING COMPLETE. RUNNING EVALUATION..."
echo "============================================================"

poetry run python eval.py
poetry run python aggregate_results.py

# --- 7. Build & evaluate ensemble (per selected mode) ---
echo ""
echo "============================================================"
echo "BUILDING MODEL ENSEMBLE..."
echo "============================================================"

for mode in "${selected_modes[@]}"; do
    echo ""
    echo "--- Ensemble: $mode ---"
    TRAINING_MODE=$mode poetry run python ensemble.py build "$mode"
    TRAINING_MODE=$mode poetry run python ensemble.py eval  "$mode"
done

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE."
echo "============================================================"