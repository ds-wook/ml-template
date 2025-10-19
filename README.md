# ml-template
machine learning template 

# Setting
We use [uv](https://github.com/astral-sh/uv) to manage dependencies of this repository.

### Install uv
You can install `uv` using different methods depending on your operating system or preference.

```bash
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

### Create virtual enviroment
For an existing project, simply sync the environment:
```
$ uv sync
```

### Run
`uv run` allows you to execute Python scripts inside the managed virtual environment.

#### train
```sh
uv run python src/train.py \
    data=boosting \
    models=lightgbm
```

#### inference
```sh
uv run python src/predict.py \
    data=boosting \
    models=lightgbm
```