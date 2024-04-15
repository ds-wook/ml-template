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

## Project Organization
```
├── LICENSE
├── README.md
├── config
│   ├── data
│   │   └── dataset.yaml
│   ├── ensemble.yaml
│   ├── experiment
│   │   └── wandb.yaml
│   ├── generator
│   │   └── featurize.yaml
│   ├── models
│   │   ├── catboost.yaml
│   │   ├── lightgbm.yaml
│   │   ├── tabnet.yaml
│   │   └── xgboost.yaml
│   ├── predict.yaml
│   └── train.yaml
├── docs
├── environment.yaml
├── input
├── notebooks
├── output
├── pyproject.toml
├── reports
│   └── figures
├── resources
│   ├── encoder
│   └── models
├── scripts
│   └── run.sh
└── src
    ├── data
    │   ├── __init__.py
    │   └── dataset.py
    ├── ensemble.py
    ├── generator
    │   ├── __init__.py
    │   ├── base.py
    │   └── featurize.py
    ├── models
    │   ├── __init__.py
    │   ├── base.py
    │   ├── nn
    │   │   └── tabnet.py
    │   └── tree
    │       └── boosting.py
    ├── predict.py
    ├── train.py
    └── utils
        ├── __init__.py
        ├── plot.py
        └── utilies.py
```
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
