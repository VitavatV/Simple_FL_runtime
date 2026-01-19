"""
Advanced benchmarking script for Federated Learning
Provides detailed profiling and comparison metrics
"""

import torch
import torch.nn as nn
from fl_train import MNISTModel, CIFAR10Model, FLServer, FLClient
from fl_train import load_mnist_data, load_cifar10_data
import time
import json
from typing import Dict, List
import numpy as np


class FLBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_device(
        self,
        dataset_name: str,
        model_class: nn.Module,
        load_data_fn,
        device: torch.device,
        device_name: str,
        num_clients: int = 10,
        num_rounds: int = 5,
        local_epochs: int = 2
    ) -> Dict:
        """Benchmark FL training on a specific device"""
        
        print(f"\nBenchmarking {dataset_name} on {device_name}...")
        
        # Load data
        client_data = load_data_fn(num_clients)
        
        # Initialize
        server = FLServer(model_class(), device)
        clients = [FLClient(i, model_class(), device) for i in range(num_clients)]
        
        # Timing components
        timings = {
            'round_times': [],
            'local_train_times': [],
            'aggregation_times': [],
            'eval_times': []
        }
        
        # Training loop with detailed timing
        total_start = time.time()
        accuracies = []
        
        for round_num in range(num_rounds):
            # Get model to clients
            server_weights = server.get_weights()
            for client in clients:
                client.set_weights(server_weights)
            
            # Local training
            train_start = time.time()
            client_weights = []
            for client_id, client in enumerate(clients):
                train_loader, _ = client_data[client_id]
                client.train(train_loader, epochs=local_epochs)
                client_weights.append(client.get_weights())
            local_train_time = time.time() - train_start
            timings['local_train_times'].append(local_train_time)
            
            # Aggregation
            agg_start = time.time()
            averaged_weights = server.aggregate_weights(client_weights)
            server.update_model(averaged_weights)
            agg_time = time.time() - agg_start
            timings['aggregation_times'].append(agg_time)
            
            # Evaluation
            eval_start = time.time()
            _, test_loader = client_data[0]
            test_client = FLClient(999, server.model, device)
            test_client.set_weights(averaged_weights)
            accuracy = test_client.evaluate(test_loader)
            eval_time = time.time() - eval_start
            timings['eval_times'].append(eval_time)
            accuracies.append(accuracy)
            
            round_time = local_train_time + agg_time + eval_time
            timings['round_times'].append(round_time)
            
            print(f"  Round {round_num + 1}/{num_rounds}: {round_time:.3f}s - Accuracy: {accuracy:.2f}%")
        
        total_time = time.time() - total_start
        
        # Compute statistics
        return {
            'device': device_name,
            'dataset': dataset_name,
            'total_time': total_time,
            'avg_round_time': np.mean(timings['round_times']),
            'avg_local_train_time': np.mean(timings['local_train_times']),
            'avg_aggregation_time': np.mean(timings['aggregation_times']),
            'avg_eval_time': np.mean(timings['eval_times']),
            'final_accuracy': accuracies[-1],
            'accuracies': accuracies,
            'round_times': timings['round_times'],
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs
        }
    
    def run_full_benchmark(self, num_clients: int = 10, num_rounds: int = 5):
        """Run complete benchmark suite"""
        
        print("=" * 70)
        print("Federated Learning Benchmarking Suite")
        print("=" * 70)
        
        # Detect devices
        devices = []
        if torch.cuda.is_available():
            devices.append(('GPU', torch.device('cuda')))
            print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            print("\n✗ GPU Not Available")
        
        devices.append(('CPU', torch.device('cpu')))
        
        # Run benchmarks
        all_results = []
        
        for dataset_name, load_data_fn, model_class in [
            ('MNIST', load_mnist_data, MNISTModel),
            ('CIFAR10', load_cifar10_data, CIFAR10Model)
        ]:
            dataset_results = {}
            
            print(f"\n{'=' * 70}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 70}")
            
            for device_name, device in devices:
                result = self.benchmark_device(
                    dataset_name,
                    model_class,
                    load_data_fn,
                    device,
                    device_name,
                    num_clients=num_clients,
                    num_rounds=num_rounds
                )
                dataset_results[device_name] = result
                all_results.append(result)
            
            # Print summary and comparison
            self.print_summary(dataset_results)
        
        # Print final summary
        self.print_final_summary(all_results)
        
        return all_results
    
    def print_summary(self, dataset_results: Dict):
        """Print summary for a dataset"""
        
        print(f"\n{'-' * 70}")
        print("Summary")
        print(f"{'-' * 70}")
        
        for device_name, result in dataset_results.items():
            print(f"\n{device_name}:")
            print(f"  Total Time:           {result['total_time']:.2f}s")
            print(f"  Avg Round Time:       {result['avg_round_time']:.3f}s")
            print(f"  Avg Local Train Time: {result['avg_local_train_time']:.3f}s")
            print(f"  Avg Aggregation Time: {result['avg_aggregation_time']:.3f}s")
            print(f"  Avg Eval Time:        {result['avg_eval_time']:.3f}s")
            print(f"  Final Accuracy:       {result['final_accuracy']:.2f}%")
        
        # Speedup comparison
        if len(dataset_results) > 1 and 'GPU' in dataset_results and 'CPU' in dataset_results:
            gpu_time = dataset_results['GPU']['total_time']
            cpu_time = dataset_results['CPU']['total_time']
            speedup = cpu_time / gpu_time
            print(f"\n{'GPU Speedup vs CPU:':<25} {speedup:.2f}x")
    
    def print_final_summary(self, all_results: List[Dict]):
        """Print final comparison across all benchmarks"""
        
        print(f"\n{'=' * 70}")
        print("Final Comparison Summary")
        print(f"{'=' * 70}\n")
        
        # Group by dataset
        by_dataset = {}
        for result in all_results:
            dataset = result['dataset']
            if dataset not in by_dataset:
                by_dataset[dataset] = {}
            by_dataset[dataset][result['device']] = result
        
        # Print comparison table
        print(f"{'Dataset':<12} {'Device':<8} {'Total (s)':<12} {'Avg Round (s)':<15} {'Accuracy':<10}")
        print("-" * 60)
        
        for dataset, devices in by_dataset.items():
            for device, result in devices.items():
                print(f"{dataset:<12} {device:<8} {result['total_time']:<12.2f} {result['avg_round_time']:<15.3f} {result['final_accuracy']:<10.2f}%")
        
        # Print speedups
        print(f"\n{'GPU Speedup (vs CPU)':<35}")
        print("-" * 35)
        for dataset, devices in by_dataset.items():
            if 'GPU' in devices and 'CPU' in devices:
                speedup = devices['CPU']['total_time'] / devices['GPU']['total_time']
                print(f"{dataset:<35} {speedup:.2f}x")
    
    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save results to JSON file"""
        # Results would be saved here
        pass


if __name__ == '__main__':
    benchmark = FLBenchmark()
    results = benchmark.run_full_benchmark(num_clients=10, num_rounds=5)
