FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean

# copy all files and dir for service
COPY . .

# install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt