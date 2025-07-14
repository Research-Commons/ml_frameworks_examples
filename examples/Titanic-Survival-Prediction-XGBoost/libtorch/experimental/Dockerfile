FROM gcc:13 AS builder

# --- ENV VARS ---
ENV LD_LIBRARY_PATH="/app/libtorch/lib:/app/xgboost/lib:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    apt-get install -y cmake unzip curl bash libomp-dev libstdc++6 python3.11-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Download libtorch ---
RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip

# --- Clone and build XGBoost ---
RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    cmake -S . -B build && \
    cmake --build build -j 8

# --- Copy C++ sources ---
COPY CMakeLists.txt /app/CMakeLists.txt
COPY CSVLoader.hpp /app/CSVLoader.hpp
COPY main.cpp /app/main.cpp

# --- Build C++ target ---
RUN cmake -S . -B build \
 && cmake --build build -j 8

# --------------------------------------
# Runtime stage
# --------------------------------------
FROM gcc:13 as runtime

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv libomp5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/example2-libtorch-experimental /app/
COPY --from=builder /app/libtorch /app/libtorch
COPY --from=builder /app/xgboost /app/xgboost

COPY dataset.py /app/dataset.py
COPY resources/Titanic-Dataset.csv /app/resources/

RUN python3 -m venv .venv
RUN .venv/bin/pip install --no-cache-dir pandas numpy scikit-learn
RUN .venv/bin/python3 dataset.py

CMD ["bash"]
