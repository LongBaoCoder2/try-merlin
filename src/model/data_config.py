import os

DATA_FOLDER = os.getenv("DATA_FOLDER", "/try-merlin/data/")
BASE_DIR = os.getenv("BASE_DIR", "/try-merlin/")

# Define paths
PROCESSED_DATA_PATH = os.path.join(DATA_FOLDER, "processed_nvt")
RETRIEVAL_PATH = os.path.join(DATA_FOLDER, "processed/retrieval")
TRAIN_PATH = os.path.join(RETRIEVAL_PATH, "train")
VALID_PATH = os.path.join(RETRIEVAL_PATH, "valid")

# Parquet part size for loading data
PART_SIZE = "500MB"