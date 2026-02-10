FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY config.json ./config.json

ENV PYTHONPATH=/app/src

CMD ["python", "scripts/run_24_7.py"]
