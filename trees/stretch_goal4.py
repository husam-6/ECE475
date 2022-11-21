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

    # For regression...
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
    # Y = (df["median house value"] - 180000) / df["median house value"].std()
    Y = df["median house value"]
    df = df.drop("median house value", axis=1)

    # Divide data into 80% training, 20% validation and normalize output
    X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.2, random_state=123)
    y_train = (y_train - 180000) / y_train.std()
    y_test = (y_test - 180000) / y_test.std()

    # Add outputs to dataframe
    X_train["outcome"] = y_train
    X_test["outcome"] = y_test

    # basic_tree_algo(X_train, X_test, 4, calc_ls, "Least Squares", regression_tree_loss)

    # For Classification
    data = sklearn.datasets.load_breast_cancer()
    X_breast = pd.DataFrame(data["data"], columns = data["feature_names"])
    Y_breast = data["target"]

    # Divide data into 80% training, 20% validation
    X_train, X_test, y_train, y_test = train_test_split(X_breast, Y_breast, test_size=0.2, random_state=123)
    
    # Add outputs
    X_train["outcome"] = y_train
    X_test["outcome"] = y_test

    # Using cross entropy
    basic_tree_algo(X_train, X_test, 4, calc_cross_entropy, "Cross Entropy", classification_tree_loss)

    # Using Misclassification
    basic_tree_algo(X_train, X_test, 4, calc_misclass, "Misclassification", classification_tree_loss)
    
    # Using Gini Index
    basic_tree_algo(X_train, X_test, 4, calc_gini, "Gini Index", classification_tree_loss)


def basic_tree_algo(df, df_test, num_splits, split_function, name_of_loss, loss_function):
    """Function to implement basic tree algorithm"""
    features = df.columns.values[:-1]  
    root = Node(df)
    loss = []
    for s in range(num_splits):
        # Loop through each region (nodes in our tree)
        for r in get_leaf_nodes(root, []):
            best_split = 0
            best_error = float("inf")
            best_feature = ""
            # Loop through each possible feature
            if r.df.shape[0] <= 1:
                continue
            for feature in features:
                # Loop through each data point in that region as a split point
                for split_point in r.df[feature]:
                    err = calculate_split_loss(split_point, r.df[[feature, "outcome"]], feature, split_function)
                    if err < best_error:
                        best_error = err
                        best_split = split_point
                        best_feature = feature
                        # best_leaf_node = r
                        
            # Split tree based on best results
            r.split(best_feature, best_split)
            loss.append(tree_loss(root, df_test, loss_function))

    plt.plot(loss)
    plt.title(f"Validation Error using {name_of_loss}")
    plt.xlabel("Split Index")
    plt.ylabel("Validation Loss")
    plt.show()


def tree_loss(root, test_set, calculate_tree_loss):
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
            y_hat = .5
        else:
            y_hat = head.df["outcome"].mean()
        
        loss[i] = calculate_tree_loss(y_hat, item)
        i+=1
    return loss.mean()


def regression_tree_loss(y_hat, item):
    return (item["outcome"] - y_hat)**2


def classification_tree_loss(y_hat, item):
    """Function to predict output based on given tree splits"""
    s = 0.001
    return - (item["outcome"] * np.log(y_hat + s) + (1 - item["outcome"]) * np.log(1 - y_hat + s))


def calculate_split_loss(split_point, data, feature, loss_function):
    """Calculate loss of the split made"""
    # For least squares, get the proposed regions
    r1 = data[data[feature] < split_point]
    r2 = data[data[feature] >= split_point]

    # We don't want split point to be ineffective
    if (len(r1) == 0) or (len(r2) == 0):
        return 10000000

    # Calculate mean values
    c1 = r1["outcome"].mean()
    c2 = r2["outcome"].mean()
    
    return loss_function(r1, r2, c1, c2)


def calc_ls(r1, r2, c1, c2):
    """Least Squares Loss (Eq 9.13 and 9.14 in tb) """
    
    return ((r1["outcome"] - c1) ** 2).sum() + ((r2["outcome"] - c2) ** 2).sum()


def calc_cross_entropy(r1, r2, c1, c2):
    """Cross Entropy loss"""
    
    # To avoid log(0)
    s = 0.001

    return (-(r1["outcome"] * np.log(c1 + s) + (1 - r1["outcome"]) * np.log(1 - c1 + s) \
           + r2["outcome"] * np.log(c2 + s) + (1 - r2["outcome"]) * np.log(1 - c2 + s))).sum()


def calc_misclass(r1, r2, c1, c2):
    """Misclassification Error"""

    # Calculate mean values
    c1 = round(c1)
    c2 = round(c2)
    
    # Calculate loss
    total = abs(r1["outcome"] - c1).sum() + abs(r2["outcome"] - c2).sum()
    return total


def calc_gini(r1, r2, c1, c2):
    """Gini Index """
    return c1 * (1 - c1) + c2 * (1 - c2)


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