# One image for every pipeline step. Built in the cloud by `gcloud builds submit` (setup.sh).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY config ./config
COPY src ./src
# editable install keeps config/config.yaml next to the code, where load_config() looks
RUN pip install --no-cache-dir -e ".[cloud]"
