FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .

COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

RUN python setup.py build_ext --inplace
RUN python setup.py build_ui

# Default output directory inside container
VOLUME ["/app/outputs"]

ENTRYPOINT ["python", "gridsearch_docker.py"]