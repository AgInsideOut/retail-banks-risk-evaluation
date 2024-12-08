# Use Python 3.8 slim image
FROM python:3.8-slim

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Requirements and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and model
COPY app/ ./app/
COPY models/ ./models/
COPY config.py .

# Environment variables
ENV MODEL_PATH=models/model.joblib
ENV PORT=8080

# Expose port
EXPOSE 8080

# Runing the application with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.main:app