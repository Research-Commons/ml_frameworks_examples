FROM gcc:11

# Install system dependencies
RUN apt update && apt install -y python3 python3-venv cmake curl

# Create working directory
WORKDIR /workspace

# Copy files
COPY . /workspace

# Set up Python virtual environment
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    \
    # Download & extract CIFAR-10
    curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && \
    tar -xvzf cifar-10-python.tar.gz && \
    \
    # Run Python script inside venv
    python prepare_cifar10.py && \
    \
    # Clean up
    rm -rf cifar-10-batches-py cifar-10-python.tar.gz && \
    \
    # Deactivate venv (optional since RUN layer ends here)
    deactivate || true

# Clone tiny-dnn (release or latest)
RUN git clone https://github.com/tiny-dnn/tiny-dnn.git

# Build tiny-dnn as a header-only lib (optional CMake build if needed)
WORKDIR /workspace/tiny-dnn
RUN cmake . -DBUILD_EXAMPLES=ON && make -j2

# Build the C++ project
WORKDIR /workspace
RUN cmake . && make

# Run the model
CMD ["./run.sh"]
