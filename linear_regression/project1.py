"""ECE475 Frequentist ML Project 1 - Linear Regression

Husam Almanakly & Michael Bentivegna

"""

# %% Libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import linear_model
# import seaborn as sns
# sns.set()

# %% Parse data and preprocessing
def process_data(df, output):
    test_validation = df[df["train"] == "F"].drop("train", axis=1)
    training = df[df["train"] == "T"].drop("train", axis=1)

    # Normalized data 
    training = (training - training.mean()) / training.std()
    test_validation = (test_validation - test_validation.mean()) / test_validation.std()

    # Save the output vectors for training / testing (lpsa column)
    output_training = output[output["train"] == "T"].drop("train", axis=1).to_numpy()
    output_test = output[output["train"] == "F"].drop("train", axis=1).to_numpy()

    # Add column of ones for training set
    training_matrix = training.to_numpy()
    test_matrix = test_validation.to_numpy()
    ones = np.matrix([np.ones(shape=(training_matrix.shape[0],))]).T

    # Training Matrix
    training_matrix = np.hstack((ones, training_matrix))

    # Repeat for the test / validations set
    ones_test = np.array([np.ones(shape=(test_matrix.shape[0],))]).T
    test_matrix = np.hstack((ones_test, test_matrix))

    # Create validation + test set
    validation_matrix = test_matrix[:15, :]
    validation_output = output_test[:15]

    test_matrix_split = test_matrix[15:, :]
    output_test_split = output_test[15:]


# %%

def lin_reg(X, y):
    """Function to calculate the linear regression model on given data
    
    X is an (N x p+1) matrix
    
    Y is an (N x 1) column vector
    """

    b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    return b_hat


def lin_reg_ridge(X, y, lamb):
    """
    X is an (N x p) matrix
    
    Y is an (N x 1) column vector

    lambda is a scalar for the bias
    """

    b_hat = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y
    b_o = np.mean(y)

    return b_hat, b_o


def mean_squared_error(y, y_hat):
    """Function for calculating the mse"""
    mse = np.mean(np.square(y-y_hat))
    
    return mse


def test_lin_reg_ridge(X, y, b_hat, b_o):
    """Function to test Ridge linear regression"""
    y_hat = b_o + X @ b_hat
    
    return mean_squared_error(y, y_hat)


def test_lin_reg(X, y, b_hat):
    """Function to test linear regression"""
    y_hat = X @ b_hat
    
    return mean_squared_error(y, y_hat)


def create_table(b_hat, tab, cols, skip=0):
    for i, item in enumerate(cols):
        tab.append((item, b_hat[i+skip]))
    
    return tab


# %% Linear regression

def apply_models(df, output):
    test_validation = df[df["train"] == "F"].drop("train", axis=1)
    training = df[df["train"] == "T"].drop("train", axis=1)

    # Normalized data 
    training = (training - training.mean()) / training.std()
    test_validation = (test_validation - test_validation.mean()) / test_validation.std()

    # Save the output vectors for training / testing (lpsa column)
    output_training = output[output["train"] == "T"].drop("train", axis=1).to_numpy()
    output_test = output[output["train"] == "F"].drop("train", axis=1).to_numpy()

    # Add column of ones for training set
    training_matrix = training.to_numpy()
    test_matrix = test_validation.to_numpy()
    ones = np.matrix([np.ones(shape=(training_matrix.shape[0],))]).T

    # Training Matrix
    training_matrix = np.hstack((ones, training_matrix))

    # Repeat for the test / validations set
    ones_test = np.array([np.ones(shape=(test_matrix.shape[0],))]).T
    test_matrix = np.hstack((ones_test, test_matrix))

    # Create validation + test set
    validation_matrix = test_matrix[:test_matrix.shape[0] // 2, :]
    validation_output = output_test[:test_matrix.shape[0] // 2]

    test_matrix_split = test_matrix[test_matrix.shape[0] // 2:, :]
    output_test_split = output_test[test_matrix.shape[0] // 2:]

    b_hat = lin_reg(training_matrix, output_training)
    tab = [("Intercept", b_hat[0])]
    tab = create_table(b_hat, tab, cols=training.columns, skip=1)

    # Test result from linear regression
    mse = test_lin_reg(test_matrix, output_test, b_hat)

    print(f"Beta Values from Linear Regression \n {tabulate(tab)}")
    print(f"Linear Regression MSE: {mse}")

    # Ridge Regression
    lambdas = np.linspace(5, 1000, 1000)

    min_mse = float("inf")
    b_ridge_best = None
    b_o_best = None
    lamb_best = 0
    betas = np.zeros((training_matrix.shape[1] - 1, 1000))
    for i, lamb in enumerate(lambdas):
        b_ridge, b_o = lin_reg_ridge(training_matrix[:, 1:], output_training, lamb)
        res = test_lin_reg_ridge(validation_matrix[:, 1:], validation_output, b_ridge, b_o)
        betas[:, i] = b_ridge.flatten()
        if res < min_mse:
            min_mse = res
            b_ridge_best = b_ridge
            b_o_best = b_o
            lamb_best = lamb

    tab = []
    tab.append(("Intercept", b_o_best))
    tab = create_table(b_ridge_best, tab, cols=training.columns)
    print(f"Best Beta_Ridge:\n {tabulate(tab)}")
    ridge_mse = test_lin_reg_ridge(test_matrix[:, 1:], output_test, b_ridge_best, b_o_best)
    print(f"Ridge MSE: {ridge_mse}")

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.log(lambdas), betas.T)
    ax.legend(training.columns)
    ax.set_xlabel("Log(Lambda)")
    ax.set_ylabel("Coefficient")
    ax.invert_xaxis()
    ax.axvline(np.log(lamb_best), ls="--")
    ax.set_title("Ridge Regression Coefficients")
    # plt.show()


    # Lasso Model
    alpha = np.linspace(0.001, 1, 1000)
    best_score = float("-inf")
    best_parameters = {}
    betas = np.zeros((training_matrix.shape[1] - 1, 1000))
    for i, alpha2 in enumerate(alpha):
        clf = linear_model.Lasso(alpha=alpha2)
        s = clf.fit(training_matrix[:, 1:].tolist(), output_training.tolist())
        score = clf.score(validation_matrix[:, 1:], validation_output)
        betas[:, i] = clf.coef_
        if score > best_score:
            best_score = score
            best_alpha = alpha2
            best_parameters["weights"] = clf.coef_
            best_parameters["intercept"] = clf.intercept_    

    tab = []
    tab.append(("Intercept", best_parameters["intercept"]))
    tab = create_table(best_parameters["weights"], tab, cols=training.columns)
    print(f"Best Beta_Lasso:\n {tabulate(tab)}")

    fig, ax = plt.subplots(1, 1)
    ax.plot(alpha, betas.T)
    ax.legend(training.columns)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Coefficient")
    ax.invert_xaxis()
    ax.axvline(best_alpha, ls="--")
    ax.set_title("Lasso Regression Coefficients")

    plt.show()

# %% Apply first Prostate Cancer Dataset
df = pd.read_csv("prostate.txt", sep="\t")

output = df[["lpsa", "train"]]
df = df.drop(["Unnamed: 0", "lpsa"], axis=1)

apply_models(df, output)

# %% Repeat for a new dataset

df2 = pd.read_csv("Real estate.csv")

df2 = df2.drop(["No"], axis=1)
df2['train'] = "T"
dfupdate=df2.sample(50)
dfupdate.train = "F"
df2.update(dfupdate)

output2 = df2[["Y house price of unit area", "train"]]
df2.drop(["Y house price of unit area"], axis=1, inplace=True)

apply_models(df2, output2)
