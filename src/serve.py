import os
import numpy as np
import pandas as pd
import feast
import seedir as sd
from nvtabular import ColumnSchema, Schema
from nvtabular import Workflow
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.softmax_sampling import SoftmaxSampling
from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ops.unroll_features import UnrollFeatures
from merlin.systems.dag.ops.workflow import TransformWorkflow
from merlin.systems.dag.ops.faiss import QueryFaiss, setup_faiss 
from merlin.systems.dag.ops.feast import QueryFeast 
from merlin.dataloader.tf_utils import configure_tensorflow

BASE_DIR = os.environ.get("BASE_DIR", "/try-merlin/")
DATA_FOLDER = os.environ.get("DATA_FOLDER", "/try-merlin/data/")

# define feature repo path
feast_repo_path = os.path.join(BASE_DIR, "feast_repo/feature_repo/")

if not os.path.isdir(os.path.join(BASE_DIR, 'faiss_index')):
    os.makedirs(os.path.join(BASE_DIR, 'faiss_index'))
faiss_index_path = os.path.join(BASE_DIR, 'faiss_index', "index.faiss")
retrieval_model_path = os.path.join(BASE_DIR, "query_tower/")
ranking_model_path = os.path.join(BASE_DIR, "dlrm/")


item_embeddings = pd.read_parquet(os.path.join(BASE_DIR, "item_embeddings.parquet"))
setup_faiss(item_embeddings, faiss_index_path, embedding_column="output_1")

feature_store = feast.FeatureStore(feast_repo_path)
user_attributes = ["user_id"] >> QueryFeast.from_feature_view(
    store=feature_store,
    view="user_features",
    column="user_id",
    include_id=True,
)

nvt_workflow = Workflow.load(os.path.join(DATA_FOLDER, 'processed_nvt/workflow'))
user_subgraph = nvt_workflow.get_subworkflow("user")
user_features = user_attributes >> TransformWorkflow(user_subgraph)
configure_tensorflow()

topk_retrieval = int(
    os.environ.get("topk_retrieval", "100")
)
retrieval = (
    user_features
    >> PredictTensorflow(retrieval_model_path)
    >> QueryFaiss(faiss_index_path, topk=topk_retrieval)
)


item_attributes = retrieval["candidate_ids"] >> QueryFeast.from_feature_view(
    store=feature_store,
    view="item_features",
    column="candidate_ids",
    output_prefix="item",
    include_id=True,
)

item_subgraph = nvt_workflow.get_subworkflow("item")
item_features = item_attributes >> TransformWorkflow(item_subgraph)

user_features_to_unroll = [
    "user_id",
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
]

combined_features = item_features >> UnrollFeatures(
    "item_id", user_features[user_features_to_unroll]
)

ranking = combined_features >> PredictTensorflow(ranking_model_path)
top_k=10
ordering = combined_features["item_id"] >> SoftmaxSampling(
    relevance_col=ranking["click/binary_classification_task"], topk=top_k, temperature=0.00000001
)
if not os.path.isdir(os.path.join(BASE_DIR, 'poc_ensemble')):
    os.makedirs(os.path.join(BASE_DIR, 'poc_ensemble'))


request_schema = Schema(
    [
        ColumnSchema("user_id", dtype=np.int32),
    ]
)

# define the path where all the models and config files exported to
export_path = os.path.join(BASE_DIR, 'poc_ensemble')

ensemble = Ensemble(ordering, request_schema)
ens_config, node_configs = ensemble.export(export_path)

# return the output column name
outputs = ensemble.graph.output_schema.column_names
print(outputs)