from merlin.schema.tags import Tags
import merlin.models.tf as mm
import tensorflow as tf

from .model_config import QUERY_TOWER_LAYERS, DLRM_OPTIMIZER, DLRM_BATCH_SIZE, EPOCHS


# TWO TOWER MODEL
def build_two_tower_model(schema):
    """Build the Two-Tower model for item retrieval."""
    model = mm.TwoTowerModel(
        schema,
        query_tower=mm.MLPBlock(QUERY_TOWER_LAYERS, no_activation_last_layer=True),
        samplers=[mm.InBatchSampler()],
        embedding_options=mm.EmbeddingOptions(infer_embedding_sizes=True),
    )
    return model


def train_two_tower_model(model, train_data, valid_data):
    """Train the Two-Tower model."""
    model.compile(
        optimizer=DLRM_OPTIMIZER,
        run_eagerly=False,
        metrics=[mm.RecallAt(10), mm.NDCGAt(10)],
    )

    model.fit(train_data, validation_data=valid_data, batch_size=DLRM_BATCH_SIZE, epochs=EPOCHS)
    return model

def save_two_tower_model(model, output_dir):
    """Save the query tower model for future use."""
    query_tower = model.retrieval_block.query_block()
    query_tower.save(f"{output_dir}/query_tower")
    print(f"Query tower saved to {output_dir}/query_tower")


# DLRM Model

def build_dlrm_model(schema):
    """Build the DLRM model for item retrieval."""
    target_column = schema.select_by_tag(Tags.TARGET).column_names[0]

    model = mm.DLRMModel(
        schema,
        embedding_dim=64,
        bottom_block=mm.MLPBlock([128, 64]),
        top_block=mm.MLPBlock([128, 64, 32]),
        prediction_tasks=mm.BinaryClassificationTask(target_column),
    )

    return model


def train_dlrm_model(model, train_data, valid_data):
    """Train the DLRM model."""
    model.compile(optimizer="adam", run_eagerly=False, metrics=[tf.keras.metrics.AUC()])
    model.fit(train_data, validation_data=valid_data, batch_size=DLRM_BATCH_SIZE)

    return model

def save_dlrm_model(model, output_dir):
    """Save the query dlrm for future use."""
    model.save(f"{output_dir}/dlrm")
    print(f"DLRM saved to {output_dir}/dlrm")
