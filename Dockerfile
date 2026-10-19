FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/.cache \
    SERVER_HOST=0.0.0.0 \
    BASE_MODEL=/models/qwen3-8b \
    TRAINED_MODEL=/models/marvin

COPY requirements_server.txt .
RUN pip install --no-cache-dir -r requirements_server.txt

COPY config.py .
COPY core/ ./core/
COPY server.py .

EXPOSE 8000

CMD ["python", "server.py"]