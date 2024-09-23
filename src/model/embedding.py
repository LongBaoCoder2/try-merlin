import os
import nvtabular as nvt
from merlin.io import Dataset
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.workflow import TransformWorkflow
import tensorflow as tf

def prepare_model(base_dir):
    two_tower_model = tf.keras.models.load_model(os.path.join(base_dir, "query_tower"))
    return two_tower_model


def prepare_dataset(retrieval_path):
    item_features = Dataset(os.path.join(retrieval_path, "item_features", "*.parquet"))
    return item_features


def embedding_dataset(model, item_features, nvt_wkflow, base_dir):
    if model is None:
        model = prepare_model()
    
    if item_features is None:
        item_features = prepare_dataset()

    # Set up workflow for item embeddings
    workflow = nvt.Workflow(["item_id"] + (['item_id', 'item_brand', 'item_category', 'item_shop'] 
                        >> TransformWorkflow(nvt_wkflow.get_subworkflow("item"))
                        >> PredictTensorflow(model.first.item_block())))

    # Extract item embeddings
    item_embeddings = workflow.fit_transform(Dataset(item_features)).to_ddf().compute()

    # Save the item embeddings to disk
    item_embeddings.to_parquet(os.path.join(base_dir, "item_embeddings.parquet"))
    return item_embeddings