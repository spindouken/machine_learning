#!/usr/bin/env python3
import GPyOpt

def define_hyperparameter_space():
    """
    Define the hyperparameter space for Bayesian optimization.
    
    Returns:
        domain (list): List of dictionaries specifying the hyperparameter space.
    """
    domain = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
        {'name': 'dense_units', 'type': 'discrete', 'domain': (128, 256, 512)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.3, 0.7)},
        {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-3)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
    ]
    return domain
