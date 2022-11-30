"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 5
"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import NMF
from surprise import Dataset
from surprise.model_selection import cross_validate


def main():
    return

# %%

algo = NMF()
data = Dataset.load_builtin('ml-100k')

latent_var = 1000
n_factors = [5, 15, 50]
reg_pu = [0.02, 0.06, 0.36]
reg_qi = [0.02, 0.06, 0.36]
reg_bu = [0.005, 0.02, 0.12]
reg_bi = [0.005, 0.02, 0.12]



# Run 5-fold cross-validation and print results.
results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)


rmse = results["test_rmse"].mean()
rmse

# if __name__ == "__main__":
    # main()