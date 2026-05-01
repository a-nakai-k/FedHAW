# FedHAW: Federated Learning with Hypergradient-based Adaptive Weights

This repository contains the implementation of **FedHAW** and the experiments presented in the letter:

> **[Federated Learning with Hypergradient-based Online Update of Aggregation Weights]**  

FedHAW is a federated learning aggregation method that uses hypergradient descent to adaptively optimize a global step-size scalar `gamma` and per-client aggregation weights `lambda_k` on the server, without requiring a proxy dataset or additional communication overhead.

---

## Overview of Experiments

| Script | Paper Section | Description |
|---|---|---|
| `run_serverside.py` | Sec. IV-B | Accuracy/loss comparison and server-side computation time |
| `run_serverside_witherror.py` | Sec. IV-C | Robustness under stochastic client errors (server-side methods) |
| `run_clientside_witherror.py` | Sec. IV-D | Robustness under stochastic client errors (client-side methods) |

### Methods compared

**Sec. IV-B & IV-C** (server-side aggregation):
- **FedAvg** — standard weighted averaging
- **FedAdp** — angle-based adaptive weighting
- **FedLWS** — layer-wise step-size scaling
- **FedHyper-G** — hypergradient-based global step-size adaptation
- **FedLAW** — proxy-dataset-based aggregation weight learning
- **FedHAW** (proposed) — hypergradient-based adaptive weights (no proxy dataset needed)

**Sec. IV-D** (client-side regularization):
- **FedProx** — proximal regularization on each client
- **FedProx+HAW** (proposed) — FedProx local objective + HAW aggregation

### Datasets

| Dataset | Model | Notes |
|---|---|---|
| MNIST | 3-layer MLP (784→128→128→10) | Auto-downloaded via torchvision |
| Stanford Dogs | 3-layer MLP on pre-extracted 768-dim features | Pre-processed feature vectors |
| CIFAR-10 | ResNet-18 (CIFAR-adapted) | Auto-downloaded via torchvision |

---

## Requirements

```
torch >= 1.12
torchvision >= 0.13
numpy
matplotlib
```

Install with:
```bash
pip install torch torchvision numpy matplotlib
```

---

## Data Directory Structure

Place datasets under `./data/` following the structure below.
The Dirichlet concentration parameter used for partitioning is encoded in the directory name
(`alpha01` for α=0.1, `alpha1` for α=1.0).

```
data/
├── mnist/
│   └── alpha01/          # MNIST, K=10 clients, Dirichlet alpha=0.1
│       └── train/
│           ├── 0.npz
│           ├── 1.npz
│           ...
│           └── 9.npz
├── cifar10/
│   └── alpha01/          # CIFAR-10, K=10 clients, Dirichlet alpha=0.1
│       └── train/
│           └── ...
└── dogs/
    ├── dogs_alltestX.pt  # Stanford Dogs test features (shape: [N_test, 768])
    ├── dogs_alltesty.pt  # Stanford Dogs test labels
    └── alpha01/          # Stanford Dogs, K=10 clients, Dirichlet alpha=0.1
        └── train/
            └── ...
```

### File format for local training data

Each `.npz` file stores one client's training data and must contain a key `'data'`
whose value is a length-1 object array. The single element is a dict with:
- `'x'`: list of input arrays (numpy, shape depending on dataset)
- `'y'`: list of integer labels

For MNIST/CIFAR-10, inputs are normalized image tensors (shape `(C, H, W)`).  
For Stanford Dogs, inputs are pre-extracted feature vectors (shape `(768,)`).

---

## Running Experiments

The scripts automatically construct the data path as `./data/{dataset}/alpha{alpha_str}/`
based on `--dataset` and `--dir_alpha`. Use `--data_dir` to override if needed.

### Sec. IV-B — Basic comparison (MNIST)

```bash
python run_serverside.py \
    --dataset mnist \
    --dir_alpha 0.1 \
    --T 200 --E 1 --mu 0.001 --opt SGD \
    --lr_gamma 0.001 --lr_lambda 0.01 \
    --result_dir ./results/mnist_serverside
```

### Sec. IV-B — Basic comparison (CIFAR-10)

```bash
python run_serverside.py \
    --dataset cifar10 \
    --dir_alpha 0.1 \
    --T 50 --E 10 --mu 0.1 --opt SGDdecay \
    --opt_proxy Adam --muproxy 0.01 \
    --lr_gamma 1e-5 --lr_lambda 0.01 \
    --result_dir ./results/cifar10_serverside
```

### Sec. IV-B — Server-side computation time (Stanford Dogs)

```bash
python run_serverside.py \
    --dataset dogs \
    --dir_alpha 0.1 \
    --T 100 --E 10 --mu 0.00005 --opt Adam \
    --opt_proxy Adam --muproxy 0.01 \
    --lr_gamma 1e-7 --lr_lambda 1e-5 \
    --num_feature 256 \
    --measure_time \
    --result_dir ./results/dogs_serverside_time
```

### Sec. IV-C — Comparison under stochastic client errors (MNIST)

```bash
python run_serverside_witherror.py \
    --dataset mnist \
    --dir_alpha 0.1 \
    --max_error_rate 0.2 --randseed 23 \
    --T 200 --E 1 --mu 0.001 --opt SGD \
    --lr_gamma 0.0001 --lr_lambda 0.01 \
    --result_dir ./results/mnist_serverside_witherror
```

### Sec. IV-D — Client-side robustness (MNIST)

```bash
python run_clientside_witherror.py \
    --dataset mnist \
    --dir_alpha 0.1 \
    --max_error_rate 0.5 --randseed 23 \
    --T 200 --E 1 --mu 0.001 --opt SGD \
    --eta_prox 0.001 \
    --lr_gamma 0.0001 --lr_lambda 0.01 \
    --result_dir ./results/mnist_clientside_witherror
```

---

## Hyperparameter Reference

### Common parameters

| Argument | Description | Default |
|---|---|---|
| `--dataset` | Dataset (`mnist`, `cifar10`, `dogs`) | — (required) |
| `--dir_alpha` | Dirichlet alpha (determines data directory) | 0.1 |
| `--data_dir` | Override data directory | auto |
| `--K` | Number of clients | 10 |
| `--T` | Number of FL rounds | 200 |
| `--E` | Local epochs per round | 1 |
| `--batch_size` | Local mini-batch size | 64 |
| `--mu` | Local learning rate | 0.001 |
| `--opt` | Local optimizer | SGD |

### FedHAW / FedProx+HAW

| Argument | Description | Default |
|---|---|---|
| `--lr_gamma` | Hypergradient step size for `gamma` | 0.001 |
| `--lr_lambda` | Hypergradient step size for `lambda` | 0.01 |

### FedLAW

| Argument | Description | Default |
|---|---|---|
| `--Eproxy` | Proxy optimization epochs | 100 |
| `--muproxy` | Proxy optimizer learning rate | 0.01 |
| `--opt_proxy` | Proxy optimizer | SGD |

### Error model (Sec. IV-C & IV-D)

| Argument | Description | Default |
|---|---|---|
| `--max_error_rate` | Max per-client error probability | 0.2 / 0.5 |
| `--randseed` | Seed for drawing per-client error rates | 23 |

---

## Repository Structure

```
.
├── run_serverside.py             # Sec. IV-B experiments
├── run_serverside_witherror.py   # Sec. IV-C experiments
├── run_clientside_witherror.py   # Sec. IV-D experiments
└── functions/
    ├── networks.py               # Model architectures (MLP, ResNet-18, proxy networks)
    ├── datasets.py               # Data loading utilities
    ├── utils.py                  # Evaluation, plotting, and helper functions
    ├── training.py               # FL training routines (no errors)
    └── training_witherror.py     # FL training routines (with stochastic client errors)
```

