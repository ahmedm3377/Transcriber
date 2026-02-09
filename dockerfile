# 1. Use NVIDIA CUDA base (Ubuntu 24.04 based)
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# 2. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_PREFERENCE=managed \
    UV_LINK_MODE=copy \
    # This tells uv to install packages to the system path instead of a .venv
    UV_PROJECT_ENVIRONMENT="/usr/local" 

WORKDIR /app

# 3. Install uv and system dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Add deadsnakes PPA and install system packages including python3.13
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        python3.13 \
        python3.13-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a system venv at /usr/local so `uv` can use it as the project environment
RUN python3.13 -m venv /usr/local \
    && /usr/local/bin/python -m pip install --upgrade pip setuptools wheel \
    || true


# 4. Install Python 3.13 via uv
RUN uv python install 3.13

# 5. Copy config files
COPY pyproject.toml uv.lock ./

# 6. Install dependencies
# We use --no-cache to avoid storage errors during huge torch downloads
# We use --locked to ensure it matches your file EXACTLY
RUN uv sync --no-install-project --no-dev --locked --python 3.13 --no-cache

# 7. Copy your pre-downloaded model and source code
COPY ./models /app/models
COPY . /app

# 8. Final sync (installs your 'try-whisper' package itself)
RUN uv sync --no-dev --locked --python 3.13

EXPOSE 8000

# Since we installed to /usr/local, we don't need 'uv run' but it's safer to keep
CMD ["uv", "run", "python", "app.py"]