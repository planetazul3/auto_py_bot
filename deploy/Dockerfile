FROM python:3.13-slim

WORKDIR /app

ENV POETRY_VERSION=2.3.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" && \
    poetry install --only main --no-root --sync

COPY src ./src
COPY scripts ./scripts
COPY config.json ./config.json

ENV PYTHONPATH=/app/src

CMD ["python", "scripts/run_24_7.py"]
