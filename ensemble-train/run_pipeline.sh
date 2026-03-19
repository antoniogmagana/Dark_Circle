#!/bin/bash
set -e

# --- 1. Sensor Selection ---
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
echo "  1) DetectionCNN"
echo "  2) ClassificationCNN"
echo "  3) WaveformClassificationCNN"
echo "  4) ClassificationLSTM"
echo "  5) IterativeMiniRocket"
echo "  6) ALL MODELS"
echo ""
read -p "Models (space-separated) > " model_input

selected_models=()
if [[ "$model_input" == *"6"* ]] || [[ "${model_input,,}" == *"all"* ]]; then
    selected_models=("DetectionCNN" "ClassificationCNN" "WaveformClassificationCNN" "ClassificationLSTM" "IterativeMiniRocket")
else
    [[ "$model_input" == *"1"* ]] && selected_models+=("DetectionCNN")
    [[ "$model_input" == *"2"* ]] && selected_models+=("ClassificationCNN")
    [[ "$model_input" == *"3"* ]] && selected_models+=("WaveformClassificationCNN")
    [[ "$model_input" == *"4"* ]] && selected_models+=("ClassificationLSTM")
    [[ "$model_input" == *"5"* ]] && selected_models+=("IterativeMiniRocket")
fi

# --- 4. Validate ---
if [ ${#selected_sensors[@]} -eq 0 ] || [ ${#selected_modes[@]} -eq 0 ] || [ ${#selected_models[@]} -eq 0 ]; then
    echo "[!] Select at least one sensor, one mode, and one model."
    exit 1
fi

total_runs=$(( ${#selected_sensors[@]} * ${#selected_modes[@]} * ${#selected_models[@]} ))

echo ""
echo "============================================================"
echo " RUN SUMMARY"
echo "============================================================"
echo "Sensors: ${selected_sensors[*]}"
echo "Modes:   ${selected_modes[*]}"
echo "Models:  ${selected_models[*]}"
echo "Total:   ${total_runs} training runs"
echo ""
read -p "Proceed? (y/n) > " confirm
[[ "${confirm,,}" != "y" ]] && echo "Aborted." && exit 0

# --- 5. Train (sensor × mode × model) ---
echo ""
run_count=0
for sensor in "${selected_sensors[@]}"; do
    for mode in "${selected_modes[@]}"; do
        for model in "${selected_models[@]}"; do
            run_count=$((run_count + 1))
            CURRENT_RUN_ID=$(date +%Y%m%d_%H%M%S)

            echo "============================================================"
            echo "RUN ${run_count}/${total_runs}"
            echo "SENSOR: $sensor | MODE: $mode | MODEL: $model"
            echo "RUN_ID: $CURRENT_RUN_ID"
            echo "============================================================"

            RUN_ID=$CURRENT_RUN_ID \
            TRAIN_SENSOR=$sensor \
            TRAINING_MODE=$mode \
            MODEL_NAME=$model \
                poetry run python train.py

            sleep 1  # Guarantee unique RUN_IDs
        done
    done
done

# --- 6. Evaluate all models ---
echo ""
echo "============================================================"
echo "TRAINING COMPLETE. RUNNING EVALUATION..."
echo "============================================================"

poetry run python eval.py

# --- 7. Aggregate individual results ---
poetry run python aggregate_results.py

# --- 8. Build ensemble ---
echo ""
echo "============================================================"
echo "BUILDING SENSOR ENSEMBLE..."
echo "============================================================"

poetry run python ensemble.py build

# --- 9. Evaluate ensemble ---
echo ""
echo "============================================================"
echo "EVALUATING ENSEMBLE..."
echo "============================================================"

poetry run python ensemble.py eval

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE."
echo "============================================================"
