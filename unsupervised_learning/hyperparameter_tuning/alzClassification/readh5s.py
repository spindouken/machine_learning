#!/usr/bin/env python3
"""
standalone script that prints the details of every model
    from the best_models directory
"""
import os

def printDetails():
    model_files = [f for f in os.listdir('best_models') if f.endswith('.h5')]
    for model_file in model_files:
        params = model_file.replace('best_model_', '').replace('.h5', '').split('_')
        param_dict = {params[i]: params[i + 1] for i in range(0, len(params), 2)}
        deets = print(f"Model: {model_file}\nParameters: {param_dict}\n")
    return deets

if __name__ == "__main__":
    printDetails()
