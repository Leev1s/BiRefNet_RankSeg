FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/tmp/hf_cache \
    HF_MODULES_CACHE=/tmp/hf_cache/modules \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloudrun.txt ./

RUN uv pip install --system -r requirements-cloudrun.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
