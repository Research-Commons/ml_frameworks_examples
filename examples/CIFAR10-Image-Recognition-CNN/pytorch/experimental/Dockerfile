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
COPY using-simple-CNN.py /app/
COPY using-pretrained-resnet.py /app/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#make gneerated directory
RUN mkdir -p /app/generated

# Run both Python files one after another
CMD bash -c "\
  echo '🟢 Running Experiment 1 - Using a simple CNN for image classification' && python using-simple-CNN.py && echo '✅ using-simple-CNN.py run complete' && echo '' && \
  echo '🟢 Running Experiment 2 - Using a pretrained ResNet18 model' && python using-pretrained-resnet.py && echo '✅ using-pretrained-resnet.py run complete' && echo ''"

