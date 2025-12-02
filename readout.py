import os
import json
import pandas as pd
import glob

# Directory containing the result JSON files
RESULTS_DIR = "results"
OUTPUT_CSV = "metrics_summary.csv"

def main():
    # 1. Find all JSON metric files
    pattern = os.path.join(RESULTS_DIR, "*_metrics.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No JSON metrics files found in {RESULTS_DIR}/")
        return

    print(f"Found {len(files)} result files. Processing...")

    all_data = []
    all_metrics = set()  # Collect all metric names encountered

    # 2. Read and parse each JSON file
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract model name:
            # Prefer JSON "model" field; if missing, use file name
            model_name = data.get("model", os.path.basename(file_path).replace("_metrics.json", ""))
            
            print(f"model: {model_name}")
            print(f"data: {data}")

            val = data.get("validation", {}) or {}
            test = data.get("test", {}) or {}

            # Collect all metric names (JSON keys, usually uppercase)
            all_metrics |= set(val.keys()) | set(test.keys())

            row = {"Method": model_name}

            # Store metric values first (column order will be rearranged later)
            for m in set(val.keys()) | set(test.keys()):
                row[f"Val_{m}"] = val.get(m, None)
                row[f"Test_{m}"] = test.get(m, None)

            all_data.append(row)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 3. Convert to DataFrame and reorganize structure
    if not all_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(all_data)

    # ---- Key part: reorder columns (Method / Val_* / Test_*) ----
    # You can sort metrics alphabetically, or use custom order.
    metric_list = ["MAE", "RMSE", "MAPE", "R2", "Bias", "CRPS", "NLL", "PICP", "IS", "TIME"]

    # Keep only columns that exist in the DataFrame
    val_cols = [f"Val_{m}" for m in metric_list if f"Val_{m}" in df.columns]
    test_cols = [f"Test_{m}" for m in metric_list if f"Test_{m}" in df.columns]

    # Some models may lack some columns; pandas automatically fills missing values with NaN
    cols = ["Method"] + test_cols
    df = df.reindex(columns=cols)

    # Format numerical values (4 decimal places)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    # 4. Display preview
    print("\n=== Metrics Summary ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)
    
    # 5. Save as CSV
    save_path = os.path.join(RESULTS_DIR, OUTPUT_CSV)
    df.to_csv(save_path, index=False)
    print(f"\nSummary saved to: {save_path}")

if __name__ == "__main__":
    main()
