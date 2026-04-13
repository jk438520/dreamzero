import argparse
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Add value function to parquet files in a dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    parquet_files = list(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {dataset_path}/data/chunk-*")
        return
        
    for file_path in tqdm(parquet_files, desc="Adding value function to parquet files"):
        df = pd.read_parquet(file_path)
        if "frame_index" in df.columns:
            df["value_function"] = df["frame_index"] / len(df)
        else:
            import numpy as np
            df["value_function"] = np.arange(len(df)) / len(df)
            
        df.to_parquet(file_path)

    # Note: Update info.json features
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
            
        if "features" in info and "value_function" not in info["features"]:
            info["features"]["value_function"] = {
                "dtype": "float64",
                "shape": [1]
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            print("Updated meta/info.json with value_function feature")

if __name__ == "__main__":
    main()
