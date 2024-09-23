import os
import argparse
from datetime import datetime
from merlin.datasets.synthetic import generate_data
from merlin.datasets.ecommerce import get_aliccp
from merlin.models.utils.dataset import unique_rows_by_features
from merlin.schema.tags import Tags

def generate_synthetic_data(num_rows, output_dir):
    """Generates synthetic dataset mimicking the real Ali-CCP dataset."""
    print(f"Generating {num_rows} rows of synthetic data...")
    
    # Split into train and validation datasets with a 70-30 split
    train_raw, valid_raw = generate_data("aliccp-raw", num_rows, set_sizes=(0.7, 0.3))
    
    # Saving the datasets
    train_path = os.path.join(output_dir, "train_synthetic.parquet")
    valid_path = os.path.join(output_dir, "valid_synthetic.parquet")
    
    train_raw.to_parquet(train_path)
    valid_raw.to_parquet(valid_path)
    
    print(f"Data saved to {output_dir}:")
    print(f" - Train dataset: {train_path}")
    print(f" - Valid dataset: {valid_path}")
    
    return train_raw, valid_raw

def load_real_data(raw_data_path, output_dir):
    """
    Loads real Ali-CCP data from raw CSV files and converts it to parquet.
    To use this function, prepare the data by following these steps:
    1. Download the raw data from
    [tianchi.aliyun.com](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408#1).
    2. Unzip the raw data to a directory.
    """
    print("Loading real Ali-CCP data from CSV files...")
    
    # Convert CSV files into parquet format
    train_raw, valid_raw = get_aliccp(raw_data_path, output_dir=output_dir)
    
    print(f"Real data loaded and saved to {output_dir}:")
    print(f" - Train dataset: {train_raw}")
    print(f" - Valid dataset: {valid_raw}")
    
    return train_raw, valid_raw

def extract_features(raw_dataset, feature_repo_path):
    """Extracts user and item features from the raw data."""
    
    # Extract unique user features
    user_features = (
        unique_rows_by_features(raw_dataset, Tags.USER, Tags.USER_ID)
        .compute()
        .reset_index(drop=True)
    )
    user_features["datetime"] = datetime.now()
    user_features["datetime"] = user_features["datetime"].astype("datetime64[ns]")
    user_features["created"] = datetime.now()
    user_features["created"] = user_features["created"].astype("datetime64[ns]")
    
    user_features_path = os.path.join(feature_repo_path, "data", "user_features.parquet")
    user_features.to_parquet(user_features_path)
    print(f"User features saved to {user_features_path}")
    
    # Extract unique item features
    item_features = (
        unique_rows_by_features(raw_dataset, Tags.ITEM, Tags.ITEM_ID)
        .compute()
        .reset_index(drop=True)
    )
    item_features["datetime"] = datetime.now()
    item_features["datetime"] = item_features["datetime"].astype("datetime64[ns]")
    item_features["created"] = datetime.now()
    item_features["created"] = item_features["created"].astype("datetime64[ns]")
    
    item_features_path = os.path.join(feature_repo_path, "data", "item_features.parquet")
    item_features.to_parquet(item_features_path)
    print(f"Item features saved to {item_features_path}")

    return user_features, item_features

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and extract features for Recommender System.")
    
    parser.add_argument(
        "--use_synthetic", 
        action="store_true", 
        help="Flag to generate a synthetic dataset instead of using real data."
    )
    parser.add_argument(
        "--num_rows", 
        type=int, 
        default=100_000, 
        help="Number of rows to generate for the synthetic dataset (default: 100,000)."
    )
    parser.add_argument(
        "--raw_data_path", 
        type=str, 
        default="/workspace/data/", 
        help="Path to raw CSV data files for the real dataset."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/workspace/output/", 
        help="Directory to save the processed datasets (default: /workspace/output/)."
    )
    parser.add_argument(
        "--feature_repo_path", 
        type=str, 
        default="/workspace/features/", 
        help="Directory to save user and item features (default: /workspace/features/)."
    )
    
    args = parser.parse_args()

    # Prepare paths for data and feature store
    DATA_FOLDER = os.environ.get("DATA_FOLDER", args.raw_data_path)
    OUTPUT_DIR = os.environ.get("BASE_DIR", args.output_dir)
    FEATURE_REPO_PATH = os.environ.get("FEATURE_REPO_PATH", args.feature_repo_path)
    
    # Ensure output and feature directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FEATURE_REPO_PATH, exist_ok=True)
    
    if args.use_synthetic:
        # Generate synthetic data
        train_raw, _ = generate_synthetic_data(args.num_rows, OUTPUT_DIR)
    else:
        # Load real Ali-CCP data
        train_raw, _ = load_real_data(DATA_FOLDER, OUTPUT_DIR)
    
    # Extract and save user and item features
    extract_features(train_raw, FEATURE_REPO_PATH)

if __name__ == "__main__":
    main()
