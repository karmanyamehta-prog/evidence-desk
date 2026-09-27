# Prepared only. Building does not authorize a public deployment.
FROM python:3.12.11-slim
WORKDIR /app
ENV DATA_DIR=/app/data HF_HOME=/app/.cache/huggingface LLM_PROVIDER=extractive \
    API_URL=http://127.0.0.1:8000 MAX_UPLOAD_MB=5 MAX_PAGES=60 DEMO_READ_ONLY=1
COPY . .
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir . \
    && useradd -m -u 1000 app && mkdir -p /app/data /app/.cache && chown -R app:app /app
USER app
EXPOSE 7860
CMD ["python", "deploy/spaces/start.py"]
