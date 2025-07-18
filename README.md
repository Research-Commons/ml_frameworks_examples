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

Use the `build-examples.sh` script. Here is an example:

```bash 
./build-examples.sh --framework libtorch --usecase 1 --copy-resources assets/
```

>Optional : add **"--no-cache"** if you want to build the docker image without caching

This will build the `experimental` version of the image by default and provide information on how to run the image.

> [!NOTE]
> 
> Docker containers are expected to run with these certain specifications:
> 
> ```bash 
> docker run --cpus="2.0" --memory="4g" --memory-swap="4g" -it <container-id>
> ```
> 
