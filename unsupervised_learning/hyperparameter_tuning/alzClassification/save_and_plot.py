#!/usr/bin/env python3
import matplotlib.pyplot as plt
import GPyOpt
import numpy as np

def save_and_plot(optimizer):
    """
    save photos of and print evaluations and plot convergence
    """
    # Plot and save convergence
    optimizer.plot_convergence()
    plt.savefig('convergence.png')

    # Plot acquisition function
    optimizer.plot_acquisition()
    plt.savefig('acquisition_function.png')

    # Save optimization evaluations to a text file
    with open('bayes_opt_MRIalz.txt', 'w') as f:
        f.write(str(optimizer.get_evaluations()))
        best_params_so_far = optimizer.X[np.argmin(optimizer.Y)]
        f.write(f"\nBest parameters so far: {best_params_so_far}\n")
        f.write(f"Mean objective: {np.mean(optimizer.Y)}\n")
        f.write(f"Standard Deviation: {np.std(optimizer.Y)}\n")
