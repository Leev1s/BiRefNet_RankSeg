---
title: BiRefNet Demo
emoji: 👁
colorFrom: purple
colorTo: green
sdk: gradio
python_version: 3.12.12
sdk_version: 5.35.0
app_file: app.py
pinned: false
license: mit
models:
- ZhengPeng7/BiRefNet
- ZhengPeng7/BiRefNet_HR
- ZhengPeng7/BiRefNet_HR-matting
- ZhengPeng7/BiRefNet-matting
- ZhengPeng7/BiRefNet-portrait
- ZhengPeng7/BiRefNet_lite
preload_from_hub:
- ZhengPeng7/BiRefNet
- ZhengPeng7/BiRefNet_HR
- ZhengPeng7/BiRefNet_HR-matting
- ZhengPeng7/BiRefNet-matting
- ZhengPeng7/BiRefNet-portrait
- ZhengPeng7/BiRefNet_lite
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Deploy to Google Cloud Run with GPU

This repo now includes a `Dockerfile` and `requirements-cloudrun.txt` for Cloud Run GPU deployment.

Build the image:

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/birefnet-gradio
```

Deploy with an NVIDIA L4 GPU:

```bash
gcloud run deploy birefnet-gradio \
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/birefnet-gradio \
  --region REGION \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --cpu 4 \
  --memory 16Gi \
  --concurrency 1 \
  --timeout 3600 \
  --max-instances 1 \
  --port 7860 \
  --allow-unauthenticated
```

Recommended optional flags for better cold-start behavior:

```bash
--cpu-boost \
--min-instances 0
```
