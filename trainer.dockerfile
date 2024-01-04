# Base image FOR MAC
FROM --platform=linux/amd64 python:3.10-slim
# FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY dtu_mlops_code/ dtu_mlops_code/
COPY data/ data/
# COPY Makefile/ data/

WORKDIR /
RUN pip install . --no-cache-dir #(1)
# !!!!!!!!!!!!!!
# the "u" here makes sure that any output from our script e.g. any print(...) statements gets redirected to our terminal. If not included you would need to use docker logs to inspect your run
ENTRYPOINT ["python", "-u", "dtu_mlops_code/train_model.py",  "train",  "--lr", "1e-4",  "--batch_size", "64", "--num_epochs", "3"]

# docker build -f trainer.dockerfile . -t trainer:latest
# docker run --name experiment1 -v $(pwd)/models:/models/ -v $(pwd)/reports/:/reports/ trainer:latest