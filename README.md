# financial-ml

Working Directory for Financial Machine Learning Research in Supervised and Reinforcement Learning Approaches in Various Asset Markets.

**For each model you submit, you should always include an evaluation (preferably a Jupyter notebook that shows some sort of visualization like a graph or table).**s

> Note that the project is built mainly on PyTorch, and new submissions are preferred to follow. However, TensorFlow still remains an open option.

## File structure

Before you push any code, please follow this structure to avoid conflicts and for easy navigation.

```plaintext
financial-ml/
│
├── src/ # All models, trigger scripts, config, etc.
│
├── test/ # Test cases for the codebase (ideally)
|
├── data/ # Upload your dataset here
|
├── models/ # Please set all your scripts to save models here
|
├── utils/ # helper functions, visualization, envs
|
└── setenv.sh # source this to run scripts
```

## Package Dependencies

If you are using conda, you can quickly setup the environment by running:

```
conda create -f environment.yml
```

_We have defaulted the name of the venv to `myenv` but you can modify it in the file._

For those using pip with your own activated venv, run:

```
pip install -r requirements.txt
```

## Absolute Imports

To resolve file import conflicts, we sometimes use absolute import. It requires you to run the following setup commands (in a bash shell) for every terminal session from the root (financial-ml) directory:

```
source setenv.sh
```

This is a very inefficient way of working and should be replaced by a workaround in the future. If you run into any file path errors, it is likely caused by the non-ideal repository structure of the project. As the modularisation process progresses we expect improvements.
