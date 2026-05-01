import glob
import os
import re

import pandas as pd


def parse_report_file(filepath):
    """
    Reads an evaluation_report.txt file and extracts the key metrics.
    """
    data = {
        "Timestamp": None,
        "Mode": None,
        "Model": None,
        "Sensor": None,
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1-Score": None,
        "MCC": None,
        "ROC-AUC": None,
        "Latency_ms": None,
        "FAR": None,
    }

    try:
        with open(filepath) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # BACKWARDS COMPATIBILITY: Look for old and new labels
            if line.startswith("Run Directory:") or line.startswith("Timestamp:"):
                data["Timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("Mode:"):
                parts = line.split("|")
                data["Mode"] = parts[0].split(":")[1].strip()
                data["Model"] = parts[1].split(":")[1].strip()
            elif line.startswith("Accuracy:"):
                data["Accuracy"] = float(line.split(":")[1].strip())
            elif line.startswith("Precision:"):
                data["Precision"] = float(line.split(":")[1].strip())
            elif line.startswith("Recall:"):
                data["Recall"] = float(line.split(":")[1].strip())
            elif line.startswith("F1-Score:"):
                data["F1-Score"] = float(line.split(":")[1].strip())
            elif line.startswith("MCC:"):
                data["MCC"] = float(line.split(":")[1].strip())
            elif line.startswith("ROC-AUC:") and "nan" not in line.lower():
                data["ROC-AUC"] = float(line.split(":")[1].strip())
            elif line.startswith("Latency:"):
                val = re.search(r"([0-9.]+)", line.split(":")[1])
                if val:
                    data["Latency_ms"] = float(val.group(1))
            elif line.startswith("False Alarm Rate:"):
                val = re.search(r"([0-9.]+)", line.split(":")[1])
                if val:
                    # Convert % back to decimal
                    data["FAR"] = float(val.group(1)) / 100.0

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

    return data


def _sensor_from_path(filepath):
    """
    Infer the sensor name from the report file path.
    Per-sensor:     saved_models/{mode}/{sensor}/{model}/{run_id}/report.txt → parts[-4] = sensor
    Best-ensemble:  saved_models/{mode}/best_ensemble/report.txt             → parts[-2] = 'best_ensemble'
    """
    parts = os.path.normpath(filepath).split(os.sep)
    # Check for best_ensemble (3-level deep: saved_models/{mode}/best_ensemble/report.txt)
    if len(parts) >= 2 and parts[-2] == "best_ensemble":
        return "best_ensemble"
    # Per-sensor: saved_models/{mode}/{sensor}/{model}/{run_id}/report.txt
    if len(parts) >= 4:
        return parts[-4]
    return "unknown"


def main():
    print("Scanning 'saved_models/' for evaluation reports...")

    # Per-sensor: saved_models/{mode}/{sensor}/{model}/{run_id}/
    sensor_pattern = os.path.join("saved_models", "*", "*", "*", "*", "evaluation_report.txt")
    # Best-ensemble: saved_models/{mode}/best_ensemble/
    ensemble_pattern = os.path.join("saved_models", "*", "best_ensemble", "evaluation_report.txt")
    report_files = glob.glob(sensor_pattern) + glob.glob(ensemble_pattern)

    if not report_files:
        print("No evaluation reports found! " "Make sure eval.py has finished running.")
        return

    all_results = []
    for filepath in report_files:
        result = parse_report_file(filepath)
        if result and result["Model"] is not None:
            result["Sensor"] = _sensor_from_path(filepath)
            all_results.append(result)

    if not all_results:
        print("Failed to parse valid data from the found reports.")
        return

    df = pd.DataFrame(all_results)

    # Sort by Mode, then ensemble rows to bottom, then rank by F1-Score, then MCC
    df["_is_ensemble"] = (df["Sensor"] == "best_ensemble").astype(int)
    df = df.sort_values(
        by=["Mode", "_is_ensemble", "F1-Score", "MCC"],
        ascending=[True, True, False, False],
    )
    df = df.drop(columns=["_is_ensemble"])

    # Reorder columns for the Master CSV
    columns = [
        "Mode",
        "Sensor",
        "Model",
        "Accuracy",
        "F1-Score",
        "Precision",
        "Recall",
        "MCC",
        "ROC-AUC",
        "Latency_ms",
        "FAR",
        "Timestamp",
    ]
    # Filter to columns that actually exist to prevent errors with old runs
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]

    output_csv = "./saved_models/master_evaluation_results.csv"
    os.makedirs("./saved_models", exist_ok=True)
    df.to_csv(output_csv, index=False)

    # --- CONSOLE OUTPUT ---
    print("\n" + "=" * 115)
    print(" " * 38 + "MASTER RESULTS LEADERBOARD")
    print("=" * 115)

    current_mode = ""
    printed_ensemble_sep = False
    for _index, row in df.iterrows():
        if row["Mode"] != current_mode:
            current_mode = row["Mode"]
            printed_ensemble_sep = False
            print(f"\n--- {current_mode.upper()} MODE ---")
            print(
                f"{'Model':<30} | {'Sensor':<15} | {'Acc':<6} | {'F1':<6} | {'MCC':<6} | {'AUC':<6} | {'Latency':<10} | {'FAR':<6} | {'Timestamp':<25}"
            )
            print("-" * 120)

        if row.get("Sensor") == "best_ensemble" and not printed_ensemble_sep:
            print("  --- BEST ENSEMBLE ---")
            printed_ensemble_sep = True

        far_str = f"{row.get('FAR', float('nan')):.2%}" if pd.notna(row.get("FAR")) else "N/A"
        latency_str = (
            f"{row.get('Latency_ms', float('nan')):.2f} ms"
            if pd.notna(row.get("Latency_ms"))
            else "N/A"
        )
        time_str = (
            str(row.get("Timestamp", "Unknown")) if pd.notna(row.get("Timestamp")) else "Unknown"
        )

        # Safe extraction for new metrics (in case older files didn't have them)
        acc = row.get("Accuracy", float("nan"))
        f1 = row.get("F1-Score", float("nan"))
        mcc = row.get("MCC", float("nan"))
        auc = row.get("ROC-AUC", float("nan"))
        sensor = str(row.get("Sensor", "unknown"))

        acc_str = f"{acc:.4f}" if pd.notna(acc) else "N/A"
        f1_str = f"{f1:.4f}" if pd.notna(f1) else "N/A"
        mcc_str = f"{mcc:.4f}" if pd.notna(mcc) else "N/A"
        auc_str = f"{auc:.4f}" if pd.notna(auc) else "N/A"

        print(
            f"{row['Model']:<30} | {sensor:<15} | {acc_str:<6} | {f1_str:<6} | {mcc_str:<6} | {auc_str:<6} | {latency_str:<10} | {far_str:<6} | {time_str:<25}"
        )

    print("=" * 115)
    print(f"\nSaved full results (including Precision & Recall) to: " f"{output_csv}\n")


if __name__ == "__main__":
    main()
