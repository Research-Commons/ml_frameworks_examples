# ML Frameworks Examples 

Some interesting use cases and their implementations in the following frameworks:

1. LibTorch (C++ counterpart of PyTorch)
2. PyTorch 
3. Tinygrad

## Source Tree 

The source tree follows this structure:

```text 
--- assets
    ...
--- examples 
    --- example 1 (name) 
        --- framework (name)
            --- base
                <files>
            --- experimental 
                <files>
```

Each framework's implementation (`base` or `experimental`) will have their own `Dockerfile` that can be used for compiling and seeing the output of the code.

## Building

### Using Docker 

It is highly recommended to use **Docker** to build the required implementation's image for convenience and dependencies free experience.

Use the `run-examples.sh` script. Here is an example:

```bash 
./build-examples.sh --framework libtorch --usecase 1 --resources assets/
```

This will build the `experimental` version of the image by default and provide information on how to run the image.

### Native builds 

Coming soon.
