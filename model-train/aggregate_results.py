import os
import glob
import pandas as pd
import re
from datetime import datetime

def parse_report_file(filepath):
    """
    Reads an evaluation_report.txt file and extracts the key metrics.
    """
    data = {
        "Timestamp": None,
        "Mode": None,
        "Model": None,
        "Accuracy": None,
        "MCC": None,
        "ROC-AUC": None,
        "Latency_ms": None,
        "FAR": None
    }
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line.startswith("Timestamp:"):
                data["Timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("Mode:"):
                # Matches "Mode: detection | Model: DetectionCNN"
                parts = line.split("|")
                data["Mode"] = parts[0].split(":")[1].strip()
                data["Model"] = parts[1].split(":")[1].strip()
            elif line.startswith("Accuracy:"):
                data["Accuracy"] = float(line.split(":")[1].strip())
            elif line.startswith("MCC:"):
                data["MCC"] = float(line.split(":")[1].strip())
            elif line.startswith("ROC-AUC:"):
                data["ROC-AUC"] = float(line.split(":")[1].strip())
            elif line.startswith("Latency:"):
                # Matches "Latency:   1.2345 ms/sample"
                val = re.search(r"([0-9.]+)", line.split(":")[1])
                if val:
                    data["Latency_ms"] = float(val.group(1))
            elif line.startswith("False Alarm Rate:"):
                # Matches "False Alarm Rate: 1.234%"
                val = re.search(r"([0-9.]+)", line.split(":")[1])
                if val:
                    data["FAR"] = float(val.group(1)) / 100.0 # Convert % back to decimal
                    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
        
    return data

def main():
    print("Scanning 'saved_models/' for evaluation reports...")
    
    # glob specifically looks for our nested folder structure
    search_pattern = os.path.join("saved_models", "*", "*", "*", "evaluation_report.txt")
    report_files = glob.glob(search_pattern)
    
    if not report_files:
        print("No evaluation reports found! Make sure eval.py has finished running.")
        return
        
    all_results = []
    for filepath in report_files:
        result = parse_report_file(filepath)
        if result and result["Model"] is not None:
            all_results.append(result)
            
    if not all_results:
        print("Failed to parse valid data from the found reports.")
        return
        
    # Convert to a Pandas DataFrame for easy sorting and formatting
    df = pd.DataFrame(all_results)
    
    # Sort the dataframe so it's grouped by Mode, and then ranked by MCC (highest first)
    df = df.sort_values(by=["Mode", "MCC"], ascending=[True, False])
    
    # Reorder columns for readability
    columns = ["Mode", "Model", "Accuracy", "MCC", "ROC-AUC", "Latency_ms", "FAR", "Timestamp"]
    df = df[columns]
    
    # Save to a master CSV in the root directory
    output_csv = f"./saved_models/master_evaluation_results_{datetime.now()}.csv"
    df.to_csv(output_csv, index=False)
    
    # --- CONSOLE OUTPUT ---
    print("\n" + "="*110)
    print(" " * 35 + "MASTER RESULTS LEADERBOARD")
    print("="*110)
    
    current_mode = ""
    for index, row in df.iterrows():
        # Print a header every time the mode changes (detection -> category -> instance)
        if row['Mode'] != current_mode:
            current_mode = row['Mode']
            print(f"\n--- {current_mode.upper()} MODE ---")
            print(f"{'Model':<30} | {'Accuracy':<10} | {'MCC':<8} | {'AUC':<8} | {'Latency':<12} | {'FAR':<8} | {'Timestamp':<30}")
            print("-" * 88)
            
        far_str = f"{row['FAR']:.2%}" if pd.notna(row['FAR']) else "N/A"
        latency_str = f"{row['Latency_ms']:.2f} ms" if pd.notna(row['Latency_ms']) else "N/A"
        
        print(f"{row['Model']:<30} | {row['Accuracy']:<10.4f} | {row['MCC']:<8.4f} | {row['ROC-AUC']:<8.4f} | {latency_str:<12} | {far_str:<8} | {row['Timestamp']:<30}")
        
    print("="*110)
    print(f"\nSaved full results to: {output_csv}\n")

if __name__ == "__main__":
    main()