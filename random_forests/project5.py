"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 5

This project aims to compare the convergence rate of a random forest regression model and a gradient
boosted tree model. In the California housing dataset, the random forests method plateaus faster, but does not reach the
same level of accuracy as gradient boosted trees were able to. Boosting is likely slower because of its relatively
smaller tree depth and low learning rate.  Thus, random forests can be useful when timing is more important 
than efficiency.  Similar trends are seen in the Life Expectancy dataset.  However, one major difference is the bigger accuracy gap between the
two methodologies.  This can likely be attributed to the large number of features in this dataset relative to the selected
number variables at each node (m).  When playing around with a larger m value (4 and 12 instead of 2 and 6), this claim was 
corroborated and the accuracy gap shrunk. 

The other goal of this project is to compare the feature importance of each model.  The random forests model was found to have a couple
of features with high importance, with the rest being substantially less. On the other hand, the gradient boosted trees model was more balanced 
in terms of importance.  This is likely due to the fact that gradient boosted trees are able to find residual adjustments to fine tune its less 
important features whereas random forests are not.  This pattern was displayed throughout both datasets but mostly in California housing.  
"""

# %% Libraries
from tkinter import N
import pandas as pd
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, InitVar
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor


def main():
    """Main"""

    # Read in dataset, obtained from http://lib.stat.cmu.edu/datasets/
    # California Housing dataset
    labels = ["median house value", "median income", "housing median age",
            "total rooms", "total bedrooms", "population", "households",
            "latitude", "longitude"]
    df = pd.read_csv('cadata.txt', delim_whitespace=True,
                    header=None, engine='python', names = labels)

    # Add AveOccupancy feature instead of households
    df["AveOccupancy"] = df["population"] / df["households"]
    
    # Replace Rooms and Bedrooms with averages...
    df["AveRooms"] = df["total rooms"] / df["households"]
    df["AveBedrooms"] = df["total bedrooms"] / df["households"]
    df.drop(["total rooms", "total bedrooms", "households"], axis=1, inplace=True)

    # Split Data from Output
    Y = df["median house value"].to_frame()
    Y = (Y - 180000) / Y.std()
    df = df.drop("median house value", axis=1)

    # Recreate figure 15.3 on California Housing data
    regression_comparison(df, Y, 0.05, 0, 1000, "California Housing Data")


    # Second dataset: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
    life_df = pd.read_csv("life.csv")
    life_df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
    life_df = life_df.fillna(0)

    Y2 = life_df["Life expectancy "].to_frame()
    life_df.drop(["Life expectancy "], axis=1, inplace=True)
    Y2 = (Y2 - Y2.mean()) / Y2.std()

    # Recreate figure 15.3 on Life Expectancy Dataset
    regression_comparison(life_df, Y2, 0.05, 0, 1000, "Life Expectancy Data")
    return


def regression_comparison(df: pd.DataFrame, Y: pd.DataFrame, 
                 learning_rate: float,
                 alpha: int, iters: int, name_of_dataset: str):
    """Function to compare gradient boosted trees with random forest regression"""
    
    # Split data into Training and Test data (80%, 20%)
    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=123)

    # Reference: https://www.datacamp.com/tutorial/xgboost-in-python
    # For max depth 4
    lossFunc = 'mean_absolute_error'
    xg_reg_4 = xgb.XGBRegressor(objective ='reg:pseudohubererror', colsample_bytree = 0.3, learning_rate = learning_rate,
                    max_depth = 4  , alpha = alpha, n_estimators = iters, eval_metric=mean_absolute_error)
    evalset = [(X_train, y_train), (X_test, y_test)]
    xg_reg_4.fit(X_train, y_train, eval_set=evalset)
    results_4 = xg_reg_4.evals_result()
    
    # For max depth 6
    xg_reg_6 = xgb.XGBRegressor(objective ='reg:pseudohubererror', colsample_bytree = 0.3, learning_rate = learning_rate,
                                max_depth = 6  , alpha = alpha, n_estimators = iters, eval_metric=mean_absolute_error)
    evalset = [(X_train, y_train), (X_test, y_test)]
    xg_reg_6.fit(X_train, y_train, eval_set=evalset)
    results_6 = xg_reg_6.evals_result()
    
    # For m = 2
    regr_2 = RandomForestRegressor(max_features = 2, min_samples_split = 50, random_state=0, warm_start=True)
    abs_error_2 = []
    
    # For m = 6
    regr_6 = RandomForestRegressor(max_features = 6, min_samples_split = 50, random_state=0, warm_start=True)
    abs_error_6 = []
    
    # Get absolute error every 10 trees
    for i in range(1, int(iters / 10 + 1)):
        regr_2.set_params(n_estimators= i*10)
        regr_2.fit(X_train, y_train.values.ravel())
        y_hat_2 = regr_2.predict(X_test)
        abs_error_2.append(abs(y_test.to_numpy().T - y_hat_2).mean())
        
        regr_6.set_params(n_estimators=i)
        regr_6.fit(X_train, y_train.values.ravel())
        y_hat_6 = regr_6.predict(X_test)
        abs_error_6.append(abs(y_test.to_numpy().T - y_hat_6).mean())

    # Plot learning curves
    plt.figure()
    plt.plot(np.arange(1, iters + 1, 10), abs_error_2, '-o',label='RF m = 2', markerfacecolor="none", markersize=5, color="#FB8C00")
    plt.plot(np.arange(1, iters + 1, 10), abs_error_6, '-o',label='RF m = 6', markerfacecolor="none", markersize=5, color="#8BC34A")
    plt.plot(np.arange(1, iters + 1, 10), results_4['validation_1'][lossFunc][::10], '-o',label='GBM depth = 4', markerfacecolor="none", markersize=5, color="#80DEEA")
    plt.plot(np.arange(1, iters + 1, 10), results_6['validation_1'][lossFunc][::10], '-o',label='GBM depth = 6', markerfacecolor="none", markersize=5, color="#880E4F")
    plt.title(name_of_dataset)
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Average Absolute Error")
    plt.ylim([abs_error_6[-1] - .05, abs_error_6[0]])
    plt.legend()
    
    # Relative importance comparison
    importance(xg_reg_6, X_train.columns.values, "Gradient Boosted Trees")
    importance(regr_6, X_train.columns.values, "Random Forest")

def importance(xg_reg, cols, name):
    # Importance plots
    # Reference: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    sorted_features = [x for _, x in sorted(zip(xg_reg.feature_importances_, cols), reverse=True)]
    importance = sorted(xg_reg.feature_importances_, reverse=True)
    importance = importance / max(importance) * 100
    plt.figure()
    plt.barh(range(len(xg_reg.feature_importances_)), importance)
    plt.xlabel(name + " Relative Importance")
    plt.yticks(range(len(xg_reg.feature_importances_)), labels = sorted_features)
    
if __name__ == "__main__":
    main()