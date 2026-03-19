import os
import re
import glob
import json
import pandas as pd


def parse_report_file(filepath):
    """Extract key metrics from an evaluation_report.txt file."""
    data = {
        "Timestamp": None, "Mode": None, "Model": None, "Sensor": None,
        "Accuracy": None, "Precision": None, "Recall": None,
        "F1-Score": None, "MCC": None, "ROC-AUC": None,
        "Latency_ms": None, "FAR": None,
    }

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    for line in lines:
        line = line.strip()

        if line.startswith(("Run Directory:", "Timestamp:")):
            data["Timestamp"] = line.split(":", 1)[1].strip()
        elif line.startswith("Mode:"):
            parts = line.split("|")
            data["Mode"] = parts[0].split(":")[1].strip()
            data["Model"] = parts[1].split(":")[1].strip()
            # Sensor field is present in new-format reports
            if len(parts) > 2 and "Sensor" in parts[2]:
                data["Sensor"] = parts[2].split(":")[1].strip()
        elif line.startswith("Accuracy:"):
            data["Accuracy"] = float(line.split(":")[1])
        elif line.startswith("Precision:"):
            data["Precision"] = float(line.split(":")[1])
        elif line.startswith("Recall:"):
            data["Recall"] = float(line.split(":")[1])
        elif line.startswith("F1-Score:"):
            data["F1-Score"] = float(line.split(":")[1])
        elif line.startswith("MCC:"):
            data["MCC"] = float(line.split(":")[1])
        elif line.startswith("ROC-AUC:") and "nan" not in line.lower():
            data["ROC-AUC"] = float(line.split(":")[1])
        elif line.startswith("Latency:"):
            match = re.search(r"([0-9.]+)", line.split(":")[1])
            if match:
                data["Latency_ms"] = float(match.group(1))
        elif line.startswith("False Alarm Rate:"):
            match = re.search(r"([0-9.]+)", line.split(":")[1])
            if match:
                data["FAR"] = float(match.group(1)) / 100.0

    # If sensor wasn't in the report text, try to read from hyperparameters.json
    if data["Sensor"] is None:
        hp_path = os.path.join(os.path.dirname(filepath), "hyperparameters.json")
        if os.path.exists(hp_path):
            try:
                with open(hp_path) as f:
                    hp = json.load(f)
                data["Sensor"] = hp.get("TRAIN_SENSOR", "unknown")
            except Exception:
                pass

    return data if data["Model"] is not None else None


def main():
    print("Scanning 'saved_models/' for evaluation reports...")

    # Search at both old depth (mode/model/run) and new depth (mode/sensor/model/run)
    patterns = [
        os.path.join("saved_models", "*", "*", "*", "evaluation_report.txt"),
        os.path.join("saved_models", "*", "*", "*", "*", "evaluation_report.txt"),
    ]
    report_files = set()
    for pattern in patterns:
        report_files.update(glob.glob(pattern))

    if not report_files:
        print("No evaluation reports found. Run eval.py first.")
        return

    results = [r for r in map(parse_report_file, sorted(report_files)) if r]
    if not results:
        print("Failed to parse any valid reports.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(
        by=["Mode", "Sensor", "F1-Score", "MCC"],
        ascending=[True, True, False, False],
    )

    desired_columns = [
        "Mode", "Sensor", "Model", "Accuracy", "F1-Score", "Precision", "Recall",
        "MCC", "ROC-AUC", "Latency_ms", "FAR", "Timestamp",
    ]
    df = df[[c for c in desired_columns if c in df.columns]]

    output_csv = "./saved_models/master_evaluation_results.csv"
    os.makedirs("./saved_models", exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Console leaderboard
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"{'MASTER RESULTS LEADERBOARD':^130}")
    print(sep)

    current_group = ""
    for _, row in df.iterrows():
        group_key = f"{row.get('Mode', '?')} / {row.get('Sensor', '?')}"
        if group_key != current_group:
            current_group = group_key
            print(f"\n--- {current_group.upper()} ---")
            print(
                f"{'Model':<30} | {'Acc':<6} | {'F1':<6} | {'MCC':<6} | "
                f"{'AUC':<6} | {'Latency':<10} | {'FAR':<6} | {'Timestamp':<25}"
            )
            print("-" * 120)

        def fmt(val, spec=".4f"):
            return f"{val:{spec}}" if pd.notna(val) else "N/A"

        far_str = f"{row.get('FAR', float('nan')):.2%}" if pd.notna(row.get("FAR")) else "N/A"
        lat_str = f"{row.get('Latency_ms', float('nan')):.2f} ms" if pd.notna(row.get("Latency_ms")) else "N/A"
        time_str = str(row.get("Timestamp", "Unknown")) if pd.notna(row.get("Timestamp")) else "Unknown"

        print(
            f"{row['Model']:<30} | {fmt(row.get('Accuracy')):<6} | "
            f"{fmt(row.get('F1-Score')):<6} | {fmt(row.get('MCC')):<6} | "
            f"{fmt(row.get('ROC-AUC')):<6} | {lat_str:<10} | {far_str:<6} | {time_str:<25}"
        )

    print(sep)
    print(f"\nFull results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
