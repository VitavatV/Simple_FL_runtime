"""
Federated Learning training with 10 clients on MNIST and CIFAR10
Compares CPU vs GPU runtime performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from collections import OrderedDict
import time
from typing import Dict, List, Tuple
import sys


# ==================== Models ====================

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== Data Loading ====================

def load_mnist_data(num_clients: int) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """Load MNIST and split among clients"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training data among clients
    train_size = len(train_dataset)
    client_size = train_size // num_clients
    
    client_data = {}
    for client_id in range(num_clients):
        start_idx = client_id * client_size
        end_idx = start_idx + client_size if client_id < num_clients - 1 else train_size
        
        subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        train_loader = DataLoader(subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        client_data[client_id] = (train_loader, test_loader)
    
    return client_data


def load_cifar10_data(num_clients: int) -> Dict[int, Tuple[DataLoader, DataLoader]]:
    """Load CIFAR10 and split among clients"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Split training data among clients
    train_size = len(train_dataset)
    client_size = train_size // num_clients
    
    client_data = {}
    for client_id in range(num_clients):
        start_idx = client_id * client_size
        end_idx = start_idx + client_size if client_id < num_clients - 1 else train_size
        
        subset = Subset(train_dataset, list(range(start_idx, end_idx)))
        train_loader = DataLoader(subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        client_data[client_id] = (train_loader, test_loader)
    
    return client_data


# ==================== Client Training ====================

class FLClient:
    def __init__(self, client_id: int, model: nn.Module, device: torch.device):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader: DataLoader, epochs: int = 1, lr: float = 0.01) -> float:
        """Train the model on client data"""
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0.0
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def get_weights(self) -> Dict:
        """Get model weights"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights: Dict):
        """Set model weights"""
        state_dict = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)


# ==================== Federated Learning Server ====================

class FLServer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
    def aggregate_weights(self, client_weights: List[Dict]) -> Dict:
        """Average weights from all clients (FedAvg)"""
        averaged_weights = {}
        
        for key in client_weights[0].keys():
            averaged_weights[key] = torch.zeros_like(client_weights[0][key])
            
            for client_w in client_weights:
                averaged_weights[key] += client_w[key]
            
            averaged_weights[key] /= len(client_weights)
        
        return averaged_weights
    
    def update_model(self, averaged_weights: Dict):
        """Update server model with averaged weights"""
        state_dict = {k: v.to(self.device) for k, v in averaged_weights.items()}
        self.model.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict:
        """Get current model weights"""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


# ==================== Training Loop ====================

def train_fl_round(
    server: FLServer,
    clients: List[FLClient],
    client_data: Dict[int, Tuple[DataLoader, DataLoader]],
    num_rounds: int = 5,
    local_epochs: int = 2
) -> Tuple[List[float], List[float]]:
    """Execute federated learning training rounds"""
    
    round_times = []
    accuracies = []
    
    for round_num in range(num_rounds):
        round_start = time.time()
        
        # Send model to all clients
        server_weights = server.get_weights()
        for client in clients:
            client.set_weights(server_weights)
        
        # Local training on clients
        client_weights = []
        for client_id, client in enumerate(clients):
            train_loader, _ = client_data[client_id]
            client.train(train_loader, epochs=local_epochs)
            client_weights.append(client.get_weights())
        
        # Aggregate weights
        averaged_weights = server.aggregate_weights(client_weights)
        server.update_model(averaged_weights)
        
        # Evaluate on test set (using first client's test data)
        _, test_loader = client_data[0]
        test_client = FLClient(999, server.model, server.model.fc1.weight.device)
        test_client.set_weights(averaged_weights)
        accuracy = test_client.evaluate(test_loader)
        accuracies.append(accuracy)
        
        round_time = time.time() - round_start
        round_times.append(round_time)
        
        print(f"Round {round_num + 1}/{num_rounds} - Time: {round_time:.2f}s - Accuracy: {accuracy:.2f}%")
    
    return round_times, accuracies


# ==================== Main ====================

def main():
    num_clients = 10
    num_rounds = 5
    local_epochs = 2
    
    print("=" * 60)
    print("Federated Learning Training - MNIST and CIFAR10")
    print("=" * 60)
    
    # Check device availability
    has_cuda = torch.cuda.is_available()
    print(f"\nCUDA Available: {has_cuda}")
    if has_cuda:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    devices = []
    if has_cuda:
        devices.append(('GPU', torch.device('cuda')))
    devices.append(('CPU', torch.device('cpu')))
    
    # Train on both datasets and devices
    for dataset_name, load_data_fn, model_class in [
        ('MNIST', load_mnist_data, MNISTModel),
        ('CIFAR10', load_cifar10_data, CIFAR10Model)
    ]:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")
        
        # Load data
        print(f"Loading {dataset_name} data...")
        client_data = load_data_fn(num_clients)
        print(f"Data loaded for {num_clients} clients")
        
        for device_name, device in devices:
            print(f"\n{'-' * 60}")
            print(f"Training on {device_name}")
            print(f"{'-' * 60}")
            
            # Initialize server and clients
            server = FLServer(model_class(), device)
            clients = [FLClient(i, model_class(), device) for i in range(num_clients)]
            
            # Train
            start_time = time.time()
            round_times, accuracies = train_fl_round(
                server, clients, client_data,
                num_rounds=num_rounds,
                local_epochs=local_epochs
            )
            total_time = time.time() - start_time
            
            # Print results
            print(f"\n{device_name} Results:")
            print(f"  Total Time: {total_time:.2f}s")
            print(f"  Avg Round Time: {np.mean(round_times):.2f}s")
            print(f"  Final Accuracy: {accuracies[-1]:.2f}%")
            
            # Speedup comparison
            if device_name == 'GPU' and len(devices) > 1:
                cpu_time = cpu_results['total_time']
                speedup = cpu_time / total_time
                print(f"  GPU Speedup vs CPU: {speedup:.2f}x")
            else:
                device_results = {'total_time': total_time, 'accuracies': accuracies}
                if device_name == 'CPU':
                    cpu_results = device_results
    
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
