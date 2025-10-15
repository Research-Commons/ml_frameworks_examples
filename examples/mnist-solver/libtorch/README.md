# MNIST Solver

## Getting libtorch 

* Build and install LibTorch: [https://pytorch.org/cppdocs/installing.html](https://pytorch.org/cppdocs/installing.html)

```bash 
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

## Building

> [!NOTE]
> 
> You need to have the libtorch distributed shared object files in this directory!
> 

```bash 
cmake -S . -B build 
cmake --build build
```
