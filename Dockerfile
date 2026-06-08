# Steering Arena — HF Docker Space. Serves the FastAPI app (UI + API).
# No model weights are bundled: scoring runs on NDIF; only OLMo-3's config +
# tokenizer are fetched at runtime (small, public).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_home

WORKDIR /app

# Torch CPU build first (from the CPU index), then the rest of the deps.
COPY requirements.txt .
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app/ ./app/
COPY web/ ./web/
COPY data/ ./data/

# HF cache must be writable by the non-root Space user.
RUN mkdir -p /tmp/hf_home && chmod -R 777 /tmp/hf_home

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
