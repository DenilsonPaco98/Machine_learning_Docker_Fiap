FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
COPY . .

RUN pip install --upgrade pip poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

CMD ["python", "src/ecommerce_ml/train.py"]
