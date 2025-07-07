FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch pandas numpy scikit-learn

# Copy code and dataset
COPY main.py /app/main.py
COPY resources/AmesHousing.csv /app/resources/AmesHousing.csv

# Create output directory
RUN mkdir -p /app/generated

# Default command
CMD ["python", "main.py"]
