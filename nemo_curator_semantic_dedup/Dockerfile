# NeMo Curator Image Deduplication Example
# Uses CUDA 12.8 for GPU-accelerated processing
FROM anyscale/ray:2.52.0-slim-py312-cu128

# Install system dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends \
        build-essential \
        unzip \
        wget \
        curl && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
# NeMo Curator with CUDA 12 support for image processing
RUN uv pip install --system "nemo-curator[image_cuda12]"

# Additional dependencies for image downloading and processing
RUN uv pip install --system \
    loguru \
    Pillow \
    aiohttp \
    tqdm \
    pandas \
    pyarrow \
    huggingface_hub \
    transformers

# Pre-download CLIP model weights to avoid runtime downloads
# This makes job startup faster and more reliable
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
model_dir = '/home/ray/model_weights/openai/clip-vit-large-patch14'; \
os.makedirs(model_dir, exist_ok=True); \
snapshot_download('openai/clip-vit-large-patch14', local_dir=model_dir)"

# Set environment variable for model directory
ENV MODEL_DIR=/home/ray/model_weights

# For LAION runs, the input parquet (e.g., laion_subset_10m.parquet) is provided
# via working_dir upload or external storage; no dataset is baked into the image.

