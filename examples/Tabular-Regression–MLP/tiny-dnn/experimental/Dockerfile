# Use an official C++ base image
FROM ubuntu:22.04

# Set noninteractive mode to avoid UI prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    build-essential \
    libpthread-stubs0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Clone tiny-dnn (release or latest)
RUN git clone https://github.com/tiny-dnn/tiny-dnn.git

# Build tiny-dnn as a header-only lib (optional CMake build if needed)
WORKDIR /workspace/tiny-dnn
RUN cmake . -DBUILD_EXAMPLES=ON && make

# Copy your project into the image
WORKDIR /workspace
COPY . /workspace

# Build your project (assumes CMake)
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Default command (optional)
CMD ["./build/test_tinydnn"]