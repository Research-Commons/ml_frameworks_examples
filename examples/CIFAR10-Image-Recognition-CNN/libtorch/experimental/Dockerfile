FROM gcc:13 AS builder

# --- ENV VARS ---
ENV LD_LIBRARY_PATH="/app/libtorch/lib:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    apt-get install -y cmake unzip curl bash libomp-dev libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download libtorch CPU-only
RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip

# --- Stage 1: Copy C++ only ---
COPY CMakeLists.txt /app/CMakeLists.txt
COPY main.cpp /app/main.cpp
COPY LoadCIFAR-10.hpp /app/LoadCIFAR-10.hpp

RUN mkdir cifar-10-batches-bin && cd cifar-10-batches-bin && curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# --- Stage 2: Build the C++ target (cached if sources unchanged) ---
RUN cmake -S . -B build -DCMAKE_PREFIX_PATH=/app/libtorch
RUN cmake --build build -j 8

CMD ["bash"]
