#!/bin/bash

# Step 1: Initialize Feast repository
cd /try-merlin/
feast init feast_repo

# Step 2: Train the model
cd /try-merlin/src
python train.py

# Step 3: Copy feature definitions
cp /try-merlin/src/define_feature/item_features.py /try-merlin/feast_repo/feature_repo/item_features.py
cp /try-merlin/src/define_feature/user_features.py /try-merlin/feast_repo/feature_repo/user_features.py

# Step 4: Apply Feast features
cd /try-merlin/feast_repo/feature_repo/
feast apply

# Step 5: Materialize features for the specified time range
feast materialize 1995-01-01T01:01:01 2025-01-01T01:01:01

# Step 6: Serve the model
cd /try-merlin/src
python serve.py

# Step 7: Start Jupyter Lab
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''


# Step 8: Starting TIS Server
tritonserver --model-repository=/try-merlin/poc_ensemble/ --backend-config=tensorflow,version=2

