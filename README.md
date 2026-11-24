# ml-template
machine learning template

# Project structure
```
.
├── config/               # Hydra configs (train / predict / ensemble)
│   ├── data/
│   ├── models/
│   ├── features/
│   └── blends/
├── scripts/              # Entry points
│   ├── train.py
│   ├── predict.py
│   └── ensemble.py
├── src/ml_template/      # Installable package (`import ml_template`)
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── input/                # Raw data
├── resource/             # Processed data and trained models
└── notebook/
```

# Setting
We use [uv](https://github.com/astral-sh/uv) to manage dependencies of this repository.
The package is built with [hatchling](https://hatch.pypa.io/latest/), and `src/ml_template` is installed as the `ml_template` package.

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

### Create virtual environment
For an existing project, simply sync the environment.
This also installs `ml_template` in editable mode.
```bash
$ uv sync
```

### Run
`uv run` allows you to execute Python scripts inside the managed virtual environment.

#### train
```bash
$ uv run python scripts/train.py \
    data=boosting \
    models=lightgbm
```

#### inference
```bash
$ uv run python scripts/predict.py \
    data=boosting \
    models=lightgbm
```

#### ensemble
```bash
$ uv run python scripts/ensemble.py
```

### Build
```bash
$ uv build
```
