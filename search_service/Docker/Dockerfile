FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy all files and dir for service
COPY . .