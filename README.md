# financial-ml
Financial Machine Learning Research in Supervised and Reinforcement Learning Approaches in Various Asset Markets. Implementing and evaluating state of the art models proposed in research papers. Work in Progress from 2024 summer by a small team of students.

## Description

As the project is still being worked on, not all progress are merged with main branch. We currently are developing and researching three models -- CNN, DNN, and deep RL. In the future reference research papers will be added. Core models are built using PyTorch and Tensorflow, MDP process for deep RL is modelled using Gymnasium API (custom env). 

## Known Issue

_As we are updating file structure to avoid absolute import, this issue will hopefully be resolved._ For now, run the following setup commands from the root (financial-ml) directory:

```
source setenv.sh
```

This is a very inefficient way of working and should be replaced by a workaround in the future. If you run into any file path errors, it is likely caused by the non-ideal repository structure of the project.
