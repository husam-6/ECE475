"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 4

"""

# %% Libraries
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
    df = df.drop("median house value", axis=1)

    # Recreate figure 10.13 on California Housing data
    regressionXG(df, Y, 0.1, 3, 5, 800)
    return


# Normalize data
def scale_df(x: pd.DataFrame) -> pd.DataFrame:
    """Scale data passed in"""
    x = (x - x.mean()) / x.std()
    return x


def regressionXG(df: pd.DataFrame, Y: pd.DataFrame, 
                 learning_rate: float, max_depth: int,
                 alpha: int, iters: int):
    """Function to apply regression using gradient boosted trees with package XGBoost
    
    Takes in data input, output, and model parameters and produces training and test error plot
    """
    
    # Split data into Training and Test data
    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=123)

    X_train = scale_df(X_train)
    X_test = scale_df(X_test)
    y_train = scale_df(y_train)
    y_test = scale_df(y_test)


    # Reference: https://www.datacamp.com/tutorial/xgboost-in-python
    lossFunc = 'mean_absolute_error'
    xg_reg = xgb.XGBRegressor(objective ='reg:pseudohubererror', colsample_bytree = 0.3, learning_rate = learning_rate,
                    max_depth = max_depth  , alpha = alpha, n_estimators = iters, eval_metric=mean_absolute_error)

    # define the datasets to evaluate each iteration
    evalset = [(X_train, y_train), (X_test, y_test)]
    xg_reg.fit(X_train, y_train, eval_set=evalset)

    # retrieve performance metrics
    results = xg_reg.evals_result()

    # Plot learning curves
    plt.figure()
    plt.plot(results['validation_0'][lossFunc], label='train')
    plt.plot(results['validation_1'][lossFunc], label='test')
    plt.title("Training and Test Absolute Error")
    plt.xlabel("Iterations M")
    plt.ylabel("Absolute Error")
    plt.ylim([0, 1])
    plt.legend()
    

    # Reference: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    sorted_features = [x for _, x in sorted(zip(xg_reg.feature_importances_, X_train.columns.values), reverse=True)]
    importance = sorted(xg_reg.feature_importances_, reverse=True)
    importance = importance / max(importance) * 100
    plt.figure()
    plt.barh(range(len(xg_reg.feature_importances_)), importance)
    plt.xlabel("Relative Importance")
    plt.yticks(range(len(xg_reg.feature_importances_)), labels = sorted_features)
    plt.show()



if __name__ == "__main__":
    main()