import os
import nvtabular as nvt
from nvtabular.ops import Categorify, Rename, LambdaOp, TagAsUserID, TagAsItemID,  \
                          TagAsUserFeatures, TagAsItemFeatures, Dropna, AddMetadata
from merlin.dag.ops.subgraph import Subgraph
from merlin.schema.tags import Tags
from merlin.datasets.synthetic import generate_data
from merlin.datasets.ecommerce import transform_aliccp

def setup_nvt_workflow():
    """Sets up the NVTabular workflow for feature engineering."""
    
    # Define raw user and item ID columns
    user_id_raw = (
        ["user_id"]
        >> Rename(postfix='_raw')  # Rename column by adding '_raw'
        >> LambdaOp(lambda col: col.astype("int32"))  # Convert to int32
        >> TagAsUserFeatures()  # Tag as user features for easier access
    )

    item_id_raw = (
        ["item_id"]
        >> Rename(postfix='_raw')
        >> LambdaOp(lambda col: col.astype("int32"))
        >> TagAsItemFeatures()
    )
    
    # Define categorify for item categories
    item_cat = Categorify(dtype="int32")
    items = (["item_id", "item_category", "item_shop", "item_brand"] >> item_cat)

    # Create subgraph for items
    subgraph_item = Subgraph(
        "item", 
        Subgraph("items_cat", items) + 
        (items["item_id"] >> TagAsItemID()) + 
        (items["item_category", "item_shop", "item_brand"] >> TagAsItemFeatures())
    )

    # Create subgraph for users
    subgraph_user = Subgraph(
        "user",
        (["user_id"] >> Categorify(dtype="int32") >> TagAsUserID()) +
        (
            [
                "user_shops",
                "user_profile",
                "user_group",
                "user_gender",
                "user_age",
                "user_consumption_2",
                "user_is_occupied",
                "user_geography",
                "user_intentions",
                "user_brands",
                "user_categories",
            ] >> Categorify(dtype="int32") >> TagAsUserFeatures()
        )
    )

    # Define the target column
    targets = ["click"] >> AddMetadata(tags=[Tags.BINARY_CLASSIFICATION, "target"])

    # Combine user, item, and target subgraphs
    outputs = subgraph_user + subgraph_item + targets

    # Add a Dropna operation to remove rows with nulls
    outputs = outputs >> Dropna()

    # Create the workflow
    nvt_workflow = nvt.Workflow(outputs)

    return nvt_workflow

def transform_and_save_data(train_raw, valid_raw, output_path, nvt_workflow):
    """Transforms the raw dataset using NVTabular workflow and saves the results."""
    
    # Perform fit and transform on the train and valid datasets
    transform_aliccp(
        (train_raw, valid_raw), 
        output_path, 
        nvt_workflow=nvt_workflow, 
        workflow_name="workflow"
    )
    
    print(f"Processed data saved to {output_path}")

def main():
    # Setup paths and environment variables
    DATA_FOLDER = os.environ.get("DATA_FOLDER", "/workspace/data/")
    output_path = os.path.join(DATA_FOLDER, "processed_nvt")
    os.makedirs(output_path, exist_ok=True)

    # Generate synthetic data
    NUM_ROWS = os.environ.get("NUM_ROWS", 100_000)
    print(f"Generating synthetic data with {NUM_ROWS} rows...")
    train_raw, valid_raw = generate_data("aliccp-raw", int(NUM_ROWS), set_sizes=(0.7, 0.3))
    
    # Setup the NVTabular workflow for feature engineering
    nvt_workflow = setup_nvt_workflow()
    
    # Apply the workflow and save the transformed data
    transform_and_save_data(train_raw, valid_raw, output_path, nvt_workflow)

if __name__ == "__main__":
    main()
