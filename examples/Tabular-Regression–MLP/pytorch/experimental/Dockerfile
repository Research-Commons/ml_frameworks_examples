FROM python:3.10-slim

WORKDIR /app

# --- Stage 1: Install dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bash \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Install python dependencies ---
RUN pip install --no-cache-dir torch pandas numpy scikit-learn

# --- Stage 3: Copy code and dataset ---
COPY main.py /app/main.py
COPY resources/AmesHousing.csv /app/resources/AmesHousing.csv

# Create output directory
RUN mkdir -p /app/generated

CMD ["bash"]
