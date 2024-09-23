FROM nvcr.io/nvidia/merlin/merlin-tensorflow:23.12

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=""
ENV NVIDIA_DRIVER_CAPABILITIES=""
ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_DISABLE_CUDA=1

# Install dependencies
RUN pip install --upgrade pip && \
    pip install feast faiss-cpu && \
    pip uninstall cudf -y && \
    pip install -U pandas && \
    pip install -U numba dask && \
    pip install pyarrow fastparquet seedir && \
    pip uninstall dask -y && \
    pip install "dask==2023.6.0"

# Set working directory
WORKDIR /try-merlin
