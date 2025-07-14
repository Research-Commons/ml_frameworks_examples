# Use specific Python version
FROM python:3.11.13-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        bash \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy code and dataset
COPY using-DMatrix-API.py /app/
COPY using-xgboost-directly.py /app/
COPY using-xgboost-pytorch-data-pipeline.py /app/
COPY requirements.txt /app/
COPY resources/Titanic-Dataset.csv /app/resources/Titanic-Dataset.csv

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Run all three Python files one after another
CMD bash -c "\
  echo '🟢 Running Experiment 1 - Using XGBoost directly' && python using-xgboost-directly.py && echo '✅ using-xgboost-directly.py run complete' && echo '' && \
  echo '🟢 Running Experiment 2 - Using XGBoost but specify DMatrix externally' && python using-DMatrix-API.py && echo '✅ using-DMatrix-API.py run complete' && echo '' && \
  echo '🟢 Running Experiment 3 - Using XGBoost + PyTorch Data Pipeline' && python using-xgboost-pytorch-data-pipeline.py && echo '✅ using-xgboost-pytorch-data-pipeline.py run complete' && echo ''"
