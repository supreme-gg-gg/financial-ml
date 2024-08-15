# financial-ml
Working Directory for Financial Machine Learning Research in Supervised and Reinforcement Learning Approaches in Various Asset Markets.

## File structure

```plaintext
financial-ml/
│
├── src/ # Contain different models
│   ├── gdqn/
|   ├── dnn/
|   └── lstm/
│
├── tests/ # Test cases for the codebase
|
├── data/ # Upload your dataset here
|
└── utils/ # helper functions, visualization, envs
```

## A note to developers

To resolve file import conflicts, we sometimes use absolute import. It requires you to run the following setup commands (in a bash shell) for every terminal session from the root (financial-ml) directory:

```
source setenv.sh
```