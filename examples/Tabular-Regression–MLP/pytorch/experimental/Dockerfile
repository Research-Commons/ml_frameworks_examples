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
COPY using-MLP.py /app/
COPY using_ignite.py /app/
COPY using_lightning.py /app/
COPY requirements.txt /app/
COPY resources/AmesHousing.csv /app/resources/AmesHousing.csv

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /app/generated
RUN mkdir -p /app/lightning_logs

# Run all three Python files one after another
CMD bash -c "\
  echo '🟢 Running Experiment 1 - Using a MLP' && python using-MLP.py && echo '✅ using-MLP.py run complete' && echo '' && \
  echo '🟢 Running Experiment 2 - Using ignite framework' && python using_ignite.py && echo '✅ using_ignite.py run complete' && echo '' && \
  echo '🟢 Running Experiment 3 - Using pytorch lightning framework' && python using_lightning.py && echo '✅ using_lightning.py run complete' && echo ''"
