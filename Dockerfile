FROM python:3.10-slim

WORKDIR /app

ENV PYTHONHASHSEED=42
ENV TF_DETERMINISTIC_OPS=1
ENV TF_CUDNN_DETERMINISTIC=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
