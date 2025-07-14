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
COPY CSVLoader.hpp /app/CSVLoader.hpp
COPY HighFive.hpp /app/HighFive.hpp

# --- Stage 2: Build the C++ target (cached if sources unchanged) ---
RUN cmake -S . -B build -DCMAKE_PREFIX_PATH=/app/libtorch
RUN cmake --build build -j 8

# --------------------------------------
# Runtime stage
# --------------------------------------
FROM python:3.10-slim as runtime

RUN apt-get update && \
    apt-get install -y libomp5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Stage 3: Image setup ---
COPY --from=builder /app/build/example1-libtorch-experimental /app/
COPY --from=builder /app/libtorch /app/libtorch

# Copy only what may change frequently (after compilation)
COPY dataset.py /app/dataset.py
COPY resources/AmesHousing.csv /app/resources/

RUN pip install --no-cache-dir pandas numpy scikit-learn
RUN python dataset.py

CMD ["bash"]
