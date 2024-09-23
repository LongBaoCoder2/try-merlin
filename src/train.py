import os
import logging
import argparse

from data_extraction import generate_synthetic_data, load_real_data, extract_features
from data_extraction.data_transform import transform_and_save_data, setup_nvt_workflow
from model.dataset import apply_nvt_workflow, load_datasets, load_schema, setup_nvt_workflow_tt
from model.embedding import embedding_dataset
from model.model import (build_dlrm_model, build_two_tower_model, 
                         save_dlrm_model, save_two_tower_model, 
                         train_dlrm_model, train_two_tower_model)


def main(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('merlin').setLevel(logging.ERROR) 
    
    # Base directory and data folder setup
    BASE_DIR = args.base_dir
    DATA_FOLDER = args.data_folder
    NUM_ROWS = args.num_rows
    output_path = os.path.join(DATA_FOLDER, "processed_nvt")
    
    logging.info(f"Base directory: {BASE_DIR}")
    logging.info(f"Data folder: {DATA_FOLDER}")
    logging.info(f"Number of rows for data: {NUM_ROWS}")
    
    # Decide whether to use synthetic data or real data
    IS_SYNTHETIC = args.is_synthetic
    DATA_PATH = args.data_path

    logging.info("Starting data generation process")
    
    if IS_SYNTHETIC:
        # Generate synthetic data if flag is set
        logging.info("Using synthetic data")
        train_raw, valid_raw = generate_synthetic_data(NUM_ROWS, DATA_PATH)
    else:
        # Load real data otherwise
        logging.info("Using real data")
        train_raw, valid_raw = load_real_data(DATA_PATH, DATA_PATH)

    # Extract user and item features
    logging.info("Extracting features from raw data")
    feature_repo_path = os.path.join(BASE_DIR, "feast_repo/feature_repo")
    user_features, item_features = extract_features(train_raw, feature_repo_path)

    # Setup NVTabular workflow and transform data
    logging.info("Setting up NVTabular workflow")
    nvt_workflow = setup_nvt_workflow()

    logging.info("Transforming and saving processed data")
    transform_and_save_data(train_raw=train_raw, 
                            valid_raw=valid_raw, 
                            output_path=output_path, 
                            nvt_workflow=nvt_workflow)

    # Train Two-Tower model
    logging.info("Training Two-Tower model")
    output_path_retrieval = os.path.join(DATA_FOLDER, "processed/retrieval")

    # Load datasets
    train_data_tt, valid_data_tt = load_datasets(train_path=os.path.join(output_path, "train"), 
                                                 valid_path=os.path.join(output_path, "valid"))
    
    logging.info("Setting up NVTabular workflow for Two-Tower")
    nvt_workflow_tt = setup_nvt_workflow_tt(train_data=train_data_tt)
    
    logging.info("Applying NVTabular workflow to Two-Tower data")
    apply_nvt_workflow(train_data_tt, valid_data_tt, nvt_workflow_tt, output_path_retrieval)

    # Reload datasets after transformation
    train_data_tt, valid_data_tt = load_datasets(train_path=os.path.join(output_path_retrieval, "train"), 
                                                 valid_path=os.path.join(output_path_retrieval, "valid"))

    logging.info("Loading schema for Two-Tower model")
    schema = load_schema('retrieval', train_data_tt)
    train_data_tt.schema = schema
    valid_data_tt.schema = schema

    # Build, train, and save Two-Tower model
    logging.info("Building Two-Tower model")
    model_tt = build_two_tower_model(schema)
    
    logging.info("Training Two-Tower model")
    model_tt = train_two_tower_model(model_tt, train_data_tt, valid_data_tt)
    
    logging.info("Saving Two-Tower model")
    save_two_tower_model(model_tt, BASE_DIR)

    # Train DLRM model
    logging.info("Training DLRM model")
    
    # Load datasets for DLRM model
    train_data_dlrm, valid_data_dlrm = load_datasets(train_path=os.path.join(output_path, "train"), 
                                                     valid_path=os.path.join(output_path, "valid"),
                                                     part_size="500MB")
    schema = train_data_dlrm.schema

    # Build, train, and save DLRM model
    logging.info("Building DLRM model")
    dlrm_model = build_dlrm_model(schema)
    
    logging.info("Training DLRM model")
    dlrm_model = train_dlrm_model(dlrm_model, train_data_dlrm, valid_data_dlrm)
    
    logging.info("Saving DLRM model")
    save_dlrm_model(dlrm_model, BASE_DIR)

    # Embedding generation
    logging.info("Generating item embeddings using the trained Two-Tower model")
    item_embeddings = embedding_dataset(model_tt, item_features, nvt_workflow, BASE_DIR)

    logging.info("Process completed successfully: EVERYTHING OK!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Tower and DLRM models")
    
    # Command-line arguments
    parser.add_argument("--base_dir", type=str, default="/try-merlin/", 
                        help="Base directory for the project")
    parser.add_argument("--data_folder", type=str, default="/try-merlin/data/", 
                        help="Folder to store data")
    parser.add_argument("--num_rows", type=int, default=100_000, 
                        help="Number of rows for synthetic data")
    parser.add_argument("--is_synthetic", type=bool, default=True, 
                        help="Use synthetic data (True) or real data (False)")
    parser.add_argument("--data_path", type=str, default="/try-merlin/data/", 
                        help="Path to the data")

    # Parse command-line arguments
    args = parser.parse_args()

    # Start the process
    main(args)
