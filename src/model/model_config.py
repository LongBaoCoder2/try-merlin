# Two-Tower Model Architecture
QUERY_TOWER_LAYERS = [128, 64]
BATCH_SIZE = 1024 * 8
EPOCHS = 1
# TT_METRICS = ["RecallAt", ""]

# DLRM Model Architecture
EMBEDDING_DIM = 64
BOTTOM_BLOCK = [128, 64]
TOP_BLOCK = [128, 64, 32]

# Optimizer and loss function for DLRM
DLRM_OPTIMIZER = "adam"
DLRM_METRICS = ["AUC"]

# Batch size and epochs
DLRM_BATCH_SIZE = 8 * 1024
DLRM_EPOCHS = 5

