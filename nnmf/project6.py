"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 6

This project implements a basic recommendation system for a dataset of 
100k movie ratings.  In order to optimize our model, we used a grid search
for varying hyperparameter values (486 combinations in total).  The best root
mean squared error that we found was 0.955 compared to the baseline of 0.963.

Best hyperparameter values:
    - 10 epochs
    - 50 latent dimensions
    - 0.36 regularization term for users
    - 0.36 regularization term for items
    - 0.12 regulatization term for user bias
    - 0.02 regulatization term for item bias
"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import NMF
from surprise import Dataset
from surprise.model_selection import GridSearchCV

# %% Main
def main():
    # Load dataset of 100k movie ratings
    data = Dataset.load_builtin('ml-100k')

    # List of hyperparameter values we will be tuning
    n_epochs = [5, 10]
    n_factors = [5, 15, 50]
    reg_pu = [0.02, 0.06, 0.36]
    reg_qi = [0.02, 0.06, 0.36]
    reg_bu = [0.005, 0.02, 0.12]
    reg_bi = [0.005, 0.02, 0.12]


    # Set-up grid search of hyper parameters
    param_grid = {"n_epochs": n_epochs, "n_factors": n_factors, 
                "reg_pu": reg_pu, "reg_qi": reg_qi,
                "reg_bu": reg_bu, "reg_bi": reg_bi}
    
    # Create and run model
    gs = GridSearchCV(NMF, param_grid, measures=["rmse", "mae"], cv=5)
    gs.fit(data)

    # Best root mean squared error
    print(gs.best_score["rmse"])

    # Display parameters that led to the best root mean squared error 
    print(gs.best_params["rmse"])
    
    
if __name__ == "__main__":
    main()
