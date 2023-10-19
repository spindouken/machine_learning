#!/usr/bin/env python3
import os
from define_hyperparameter_space import define_hyperparameter_space
from BayesianOptimization import BayesianOptimization
from BayesianOptimization import get_best_model, get_best_hyperparameters
from load_and_preprocess import load_data
from save_and_plot import save_and_plot
import GPyOpt
import pickle
import datetime


def main():
    """
    Run Bayesian optimization to tune hyperparameters using GPyOpt
        and save the best hyperparameters (in best_params.pkl) to be used in training final_model.py
        .pkl file will come with timestamp to account for multiple bayesian optimization runs

    Main function utilizes the following functions to perform Bayesian optimization:
        define_hyperparameter_space.py
        BayesianOptimization.py
        load_and_preprocess.py
        save_and_plot.py

    Note: bayesian optimization is actually performed in BayesianOptimization.py
    """
    # create a directory for best models if it doesn't exist
    if not os.path.exists('best_models'):
        os.makedirs('best_models')

    print("Starting Bayesian optimization...")

    # use function (from main folder)
    #   which defined the hyperparameter space for Bayesian optimization
    domainExpansion = define_hyperparameter_space()

    # initialize bayesian optimization
    # add initial_design_numdata=0 to avoid random initialization
    #   and speed up optimization (for bug testing)
    optimizer = GPyOpt.methods.BayesianOptimization(
        f=BayesianOptimization,
        domain=domainExpansion,
        acquisition_type="EI",  # expected improvement
        exact_feval=True,
        maximize=False,
    )

    # specify max run count for optimization
    optimizer.run_optimization(max_iter=1)

    best_model = get_best_model()
    bestHyperparameters = get_best_hyperparameters()

    if best_model is not None:
        best_model.save(f"best_models/bestModel_{bestHyperparameters}.h5")

    print(
        "Bayesian optimization completed. Next step: Use the best hyperparameters to train your final model."
    )

    timestamp = datetime.datetime.now().strftime("%m-%d-%y-%H:%M")
    filename = f"best_params_{timestamp}.pkl"
    # retrieve best parameters from optimizer and save them to be used in final_model.py
    best_params = optimizer.x_opt
    with open(filename, "wb") as f:
        pickle.dump(best_params, f)

    # save and plot the results of the optimization
    #   this will save the convergence plot as 'convergence.png'
    #   and the optimization evaluations as 'bayes_opt_MRIalz.txt'
    save_and_plot(optimizer)
    print(
        "Best hyperparameters saved to best_params_{timestamp}.pkl. Convergence and acquisition visualizations were stored in their respective .png files."
    )

if __name__ == "__main__":
    main()
