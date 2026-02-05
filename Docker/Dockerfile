FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# copy all files and dir for service
COPY . .

# install torch cpu
RUN pip install --no-cache-dir \
    torch==2.9.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt