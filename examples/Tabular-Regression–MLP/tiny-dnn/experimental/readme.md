First run the `preprocess file`, i.e. `python3 preprocess.py`

Then setup the docker image and run the container

`docker build -t test-tinydnn .`

`docker run --rm -it test-tinydnn`