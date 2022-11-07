"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 4

    This project uses gradient boosted trees to determine a regression model for multivariate features.  The first plot
    displays the loss of the training and test sets with respect to the training time.  The next plot displays the
    relative importance of each feature.  The final two plots showcase the partial dependence with respect to one (or
    more) of the features.

    We found a dataset pertaining to life expectancy with respect to a variety of risk factors such as BMI and alcohol 
    consumption.  After initially running the model with the same hyperparameters (such as training time and
    regularization) as the textbook dataset, we fine tuned them to increase the efficacy of our model.  We first found
    that the model converged quickly even with the lower learning rate being used, so we reduced the number of
    iterations.  Next, as the number of parameters was larger than in the housing dataset, we decided to increase tree 
    depth to allow for more splits. This improved both test and training accuracy with little affect on runtime.  
    Lastly, we played around with regularization and found that an alpha value of 5 was best for
    regression on both datasets.
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
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from mpl_toolkits.mplot3d import Axes3D


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

    # Recreate figure 10.13 on California Housing data
    regressionXG(df, Y, 0.1, 3, 5, 800, 
                ['median income', 'AveOccupancy', 'housing median age', 'AveRooms'],
                ['AveOccupancy', 'housing median age'])


    # Second dataset: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
    life_df = pd.read_csv("life.csv")
    life_df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
    life_df = life_df.fillna(0)

    Y2 = life_df["Life expectancy "].to_frame()
    life_df.drop(["Life expectancy "], axis=1, inplace=True)
    Y2 = (Y2 - Y2.mean()) / Y2.std()

    regressionXG(life_df, Y2, 0.05, 4, 5, 500,
                    ['Schooling', 'infant deaths', 'Alcohol', ' BMI '],
                    ["Schooling", " BMI "])
    return


def regressionXG(df: pd.DataFrame, Y: pd.DataFrame, 
                 learning_rate: float, max_depth: int,
                 alpha: int, iters: int, features1: list,
                 features2: list):
    """Function to apply regression using gradient boosted trees with package XGBoost
    
    Takes in data input, output, and model parameters and produces training and test error plot
    """
    
    # Split data into Training and Test data
    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=123)

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

    # Relative importance 
    importanceXG(xg_reg, X_train.columns.values)
    
    # Partial dependency plots
    partial_dependencies(xg_reg, X_train, features1, features2)


def importanceXG(xg_reg, cols):
    # Importance plots
    # Reference: https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    sorted_features = [x for _, x in sorted(zip(xg_reg.feature_importances_, cols), reverse=True)]
    importance = sorted(xg_reg.feature_importances_, reverse=True)
    importance = importance / max(importance) * 100
    plt.figure()
    plt.barh(range(len(xg_reg.feature_importances_)), importance)
    plt.xlabel("Relative Importance")
    plt.yticks(range(len(xg_reg.feature_importances_)), labels = sorted_features)


def partial_dependencies(xg_reg, X_train, features1, features2):
    # Partial Dependencies
    fig, ax = plt.subplots(2, 2, figsize = (10, 10))
    PartialDependenceDisplay.from_estimator(xg_reg, X_train, features1, ax = ax)

    # Multi Var Partial Dependencies
    # Reference: https://scikit-learn.org/0.22/auto_examples/inspection/plot_partial_dependence.html
    fig = plt.figure()

    pdp = partial_dependence(xg_reg, X_train, features=features2,
                                   grid_resolution=20)
    averages = pdp['average']
    axes = pdp['values']
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = averages[0].T 
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                        cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_ylim(YY.max(), YY.min())
    ax.set_xlabel(features2[0])
    ax.set_ylabel(features2[1])
    ax.set_zlabel('Partial dependence')
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.subplots_adjust(top=0.9)

    
    plt.show()


if __name__ == "__main__":
    main()