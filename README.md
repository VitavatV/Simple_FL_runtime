# Federated Learning Runtime Comparison

PyTorch implementation of Federated Learning (FedAvg) with 10 clients trained on MNIST and CIFAR10 datasets. This project compares CPU vs GPU runtime performance.

## Features

- **10 Federated Learning Clients**: Data is split across 10 clients
- **Two Datasets**: MNIST and CIFAR10 for comprehensive benchmarking
- **CPU/GPU Comparison**: Automatically benchmarks both devices when available
- **FedAvg Algorithm**: Standard federated averaging for model aggregation
- **Detailed Timing**: Tracks round-by-round training time and total execution time

## Architecture

### Models

- **MNISTModel**: Simple 3-layer fully connected network (784 → 128 → 64 → 10)
- **CIFAR10Model**: Lightweight CNN with 2 conv layers and fully connected layers

### Federated Learning Flow

1. Server sends current model to all clients
2. Each client trains locally on their data
3. Server aggregates (averages) all client model updates
4. Repeat for multiple rounds

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fl_train.py
```

## Configuration

Modify these parameters in `fl_train.py`:

- `num_clients`: Number of federated learning clients (default: 10)
- `num_rounds`: Number of federated learning rounds (default: 5)
- `local_epochs`: Local training epochs per client per round (default: 2)
- Batch size: 32 (can be modified in data loaders)
- Learning rate: 0.01 (can be modified in client training)

## Results Interpretation

- **GPU Speedup**: Shows how much faster GPU training is compared to CPU
- **Accuracy**: Final model accuracy on test set after FL training
- **Round Time**: Time for one complete federated learning round

## System Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA 11.8+ (for GPU acceleration, optional)
- At least 4GB RAM (8GB+ recommended for CIFAR10)

## Notes

- First run downloads datasets (~200MB for MNIST, ~170MB for CIFAR10)
- GPU acceleration requires NVIDIA GPU with CUDA support
- Training times vary based on hardware specifications
- Each client gets an equal split of training data