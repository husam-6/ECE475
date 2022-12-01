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
from surprise.model_selection import GridSearchCV


def main():
    return

# %%

data = Dataset.load_builtin('ml-100k')

latent_var = 1000
n_factors = [5, 15, 50]
reg_pu = [0.02, 0.06, 0.36]
reg_qi = [0.02, 0.06, 0.36]
reg_bu = [0.005, 0.02, 0.12]
reg_bi = [0.005, 0.02, 0.12]



param_grid = {"n_epochs": [5, 10], "n_factors": n_factors, 
              "reg_pu": reg_pu, "reg_qi": reg_qi,
              "reg_bu": reg_bu, "reg_bi": reg_bi}
gs = GridSearchCV(NMF, param_grid, measures=["rmse", "mae"], cv=5)
gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])

# if __name__ == "__main__":
    # main()
# %%
