FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# copy all files and dir for service
COPY . .

# install torch cpu
RUN pip install --no-cache-dir \
    torch==2.9.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# важно: папка с моделями
# (или монтируется volume)
# /app/models/

EXPOSE 50051

CMD ["python", "-m", "service.grpc_server"]