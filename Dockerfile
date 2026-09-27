FROM python:3.12.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/app/data \
    HF_HOME=/app/.cache/huggingface ANONYMIZED_TELEMETRY=False
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY grounded_qa ./grounded_qa
COPY ui ./ui
COPY corpus ./corpus
COPY evaluation/queries.json ./evaluation/queries.json
RUN pip install --no-deps . && useradd -m -u 1000 app \
    && mkdir -p /app/data /app/.cache && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "grounded_qa.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
