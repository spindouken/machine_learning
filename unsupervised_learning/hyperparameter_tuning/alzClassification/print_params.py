#!/usr/bin/env python3
import pickle
from define_hyperparameter_space import define_hyperparameter_space  # Import the function

with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)
    
    # get hyperparameter names
    hyperparameter_space = define_hyperparameter_space()
    hyperparameter_names = [param['name'] for param in hyperparameter_space]
    
    # map names to best_params and print
    named_best_params = dict(zip(hyperparameter_names, best_params))
    
    print("Best Parameters:")
    for name, value in named_best_params.items():
        print(f"{name}: {value}")
