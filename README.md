[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
# ml-template
This is a machine learning code template utilizing the `hydra` library.
The code style has been configured to use Black, and the maximum line length has been set to 120 characters.

## Setting

The settings for the experimental environment are as follows.
- OP: Ubuntu 18.0
- CPU: i7-11799K core 8
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti


## Requirements

I conducted the experiment using `hydra-core==1.2.0` version. Please install the library based on the following information.
```sh
$ conda env create --file environment.yaml
```

## Run code
It's a code from training to inferencing.

```sh
$ sh scripts/run.sh
```
