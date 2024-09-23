from merlin.io import Dataset
import nvtabular as nvt
from nvtabular.ops import Filter



def load_datasets(train_path, valid_path, part_size=None):
    """Load training and validation datasets."""
    train_data = Dataset(f"{train_path}/*.parquet", part_size=part_size)
    valid_data = Dataset(f"{valid_path}/*.parquet", part_size=part_size)
    return train_data, valid_data

def load_schema(model, dataset: Dataset):
    """Load and filter schema for Two-Tower model."""
    if model == 'retrieval':
        schema = dataset.schema.select_by_tag(["item_id", "user_id", "item", "user"]).without(['click'])
    else:
        schema = dataset.schema
    return schema


# Define data workflow 
def setup_nvt_workflow_tt(train_data: Dataset):
    """Set up NVTabular workflow to filter clicks."""
    # Get column names and apply filter
    inputs = train_data.schema.column_names
    outputs = inputs >> Filter(f=lambda df: df["click"] == 1)

    # Create and return workflow
    nvt_wkflow = nvt.Workflow(outputs)
    return nvt_wkflow

def apply_nvt_workflow(train_data, valid_data, nvt_wkflow, output_path):
    """Apply NVTabular workflow to train and validation datasets."""
    # Fit and transform train data
    nvt_wkflow.fit(train_data)
    nvt_wkflow.transform(train_data).to_parquet(output_path=f"{output_path}/train")

    # Transform valid data
    nvt_wkflow.transform(valid_data).to_parquet(output_path=f"{output_path}/valid")


def main():
    import os

    # Load raw datasets
    train_tt, valid_tt = load_datasets()

    # Set up NVTabular workflow
    nvt_wkflow = setup_nvt_workflow_tt(train_tt)
    RETRIEVAL_PATH = os.getenv("RETRIEVAL_PATH", "/try-merlin/")
    # Apply NVTabular workflow and save the processed data
    apply_nvt_workflow(train_tt, valid_tt, nvt_wkflow, RETRIEVAL_PATH)

    print(f"Data processing complete. Processed data saved to {RETRIEVAL_PATH}")


if __name__ == "__main__":
    main()