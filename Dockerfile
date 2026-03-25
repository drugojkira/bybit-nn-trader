FROM python:3.11-slim
WORKDIR /app

# curl для healthcheck + зависимости для matplotlib
RUN apt-get update && apt-get install -y \
    curl \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Директории для данных
RUN mkdir -p /app/models /app/data /app/logs /app/data/training_logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
