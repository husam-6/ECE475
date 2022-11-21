"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 4 Stretch Goal
"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, InitVar
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from mpl_toolkits.mplot3d import Axes3D
import sklearn.datasets


def main():
    """Main"""
    
    # # Toy dataset from sklearn
    # data = sklearn.datasets.load_diabetes()
    # X_train, X_test, y_train, y_test = train_test_split(data["data"], data["target"], test_size=0.2, random_state=123)

    # # Store data in a dataframe
    # df = pd.DataFrame(X_train, columns=data["feature_names"])
    # df["outcome"] = y_train
    # df["outcome"] = (df["outcome"] - df["outcome"].mean()) / df["outcome"].std()

    # # Store test data
    # df_test = pd.DataFrame(X_test, columns=data["feature_names"])
    # df_test["outcome"] = y_test
    # df_test["outcome"] = (df_test["outcome"] - df_test["outcome"].mean()) / df_test["outcome"].std()

    # Read in dataset, obtained from http://lib.stat.cmu.edu/datasets/
    # California Housing dataset
    labels = ["median house value", "median income", "housing median age",
            "total rooms", "total bedrooms", "population", "households",
            "latitude", "longitude"]
    df = pd.read_csv('cadata.txt', delim_whitespace=True,
                    header=None, engine='python', names = labels)
    df = df.iloc[:2000]

    # Add AveOccupancy feature instead of households
    df["AveOccupancy"] = df["population"] / df["households"]
    
    # Replace Rooms and Bedrooms with averages...
    df["AveRooms"] = df["total rooms"] / df["households"]
    df["AveBedrooms"] = df["total bedrooms"] / df["households"]
    df.drop(["total rooms", "total bedrooms", "households"], axis=1, inplace=True)

    # Split Data from Output
    Y = (df["median house value"] - 180000) / df["median house value"].std()
    df = df.drop("median house value", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=123)

    X_train["outcome"] = y_train
    X_test["outcome"] = y_test

    basic_tree_algo(X_train, X_test, 4, calc_loss_ls, "Least Squares")


def basic_tree_algo(df, df_test, num_splits, loss_function, name_of_loss):
    """Function to implement basic tree algorithm"""
    features = df.columns.values[:-1]  
    root = Node(df)
    regression_loss = np.zeros(2**4 - 1)
    i = 0
    for s in range(num_splits):
        # Loop through each region (nodes in our tree)
        for r in get_leaf_nodes(root, []):
            best_split = 0
            best_error = float("inf")
            best_feature = ""
            # Loop through each possible feature
            if r.df.shape[0] == 0:
                continue
            for feature in features:
                # Loop through each data point in that region as a split point
                for split_point in r.df[feature]:
                    err = loss_function(split_point, r.df[[feature, "outcome"]], feature)
                    if err < best_error:
                        best_error = err
                        best_split = split_point
                        best_feature = feature
                        # best_leaf_node = r
                        
            # Split tree based on best results
            r.split(best_feature, best_split)
            regression_loss[i] = regression_tree_loss(root, df_test)
            i+=1

    plt.plot(regression_loss)
    plt.title(f"Validation Error using {name_of_loss}")
    plt.xlabel("Split Index")
    plt.ylabel("Validation Loss")
    plt.show()

def regression_tree_loss(root, test_set):
    """Function to predict output based on given tree splits"""
    loss = np.zeros(test_set.shape[0])
    i = 0
    for index, item in test_set.iterrows():
        head = root
        while(head.left is not None):
            if item[head.feature] < head.split_point:
                head = head.left
            else:
                head = head.right
        
        # head is the correct leaf node now
        if head.df.shape[0] == 0:
            y_hat = 0
        else:
            y_hat = head.df["outcome"].mean()

        loss[i] = (item["outcome"] - y_hat)**2
        i+=1
    
    return loss.mean()


def calc_loss_ls(split_point, data, feature):
    """Least Squares Loss (Eq 9.13 and 9.14 in tb) """

    # For least squares, get the proposed regions
    r1 = data[data[feature] < split_point]
    r2 = data[data[feature] >= split_point]

    output = "outcome"

    # Calculate mean values
    c1 = r1[output].mean()
    c2 = r2[output].mean()
    
    # Calculate loss
    total = ((r1[output] - c1) ** 2).sum() + ((r2[output] - c2) ** 2).sum()
    return total


class Node:
    """ Node class for tree algorithm """
    def __init__(self, df):
        self.left = None
        self.right = None
        self.feature = None
        self.split_point = None
        self.df = df
    
    def split(self, feature, split_point):
        self.split_point = split_point
        self.feature = feature

        self.left = Node(self.df[self.df[self.feature] < self.split_point])
        self.right = Node(self.df[self.df[self.feature] >= self.split_point])

    def printNode(self):
        print(f"Data: {self.df}")      


def get_leaf_nodes(head, arr):
    """ Function to get terminal nodes in the tree"""
    if head is None:
        return
    if head.left is None:
        arr.append(head)
    get_leaf_nodes(head.left, arr)
    get_leaf_nodes(head.right, arr)

    return arr


if __name__ == "__main__":
    main()