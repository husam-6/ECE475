"""
ECE475 Frequentist ML Project 1 - Linear Regression
Husam Almanakly & Michael Bentivegna

    This project uses three linear regression techniques (regular, ridge, and lasso) to estimate the weights of each feature in 
two unique datasets.  For regular linear regression, the covariance of the predictors is displayed as well as each features' zscore.
The ridge regression model showcases how each weight is affected by the changing lambda value which, when large, pushes all weights 
to 0.  The optimal lambda value was chosen using the validation set and is displayed on the graph with a vertical line.  The lasso model
similarly sweeps over the hyperparameter alpha to minimize MSE on the validation set.  Using lasso, the parameters lcavol, lweight, svi, 
lbph, pgg45 were all non-zero.  Although it is difficult to know if this should be the case, the lasso plot matches the results in the "The Elements 
of Statistical Learning" textbook.  For each method, the training MSE and test MSE are displayed for comparison.

Stretch Goal: 
    We attempted to add non-linear and interaction terms to the data in attempt to improve 
    the performance of our model.  Initially, each of the input data columns were plotted 
    against the output to identify features where adding a non-linear term might increase 
    linearity.  We found that pgg45 seemed to have a logarithmic relationship, so 
    we introduced an exponential term (np.exp(pgg45)). For interaction terms, we similarly
    plotted the features against themselves to look for relationships in the data. We found
    that age and lcavol had an inverse quadratic relationship (roughly speaking) and added it to our
    dataset. We plotted the new feature (- age * lcavol) against the output (lpsa)
    and we found they had a linear relationship, so it made sense that adding it would improve our 
    models loss. 

    Our baseline MSE (without any data alteration, using Ridge MSE) was about 0.74. After adding 
    the exponential term of pgg45, this decreased to 0.68.  With the introduction of interaction terms
    we were able to bring this down further to 0.64.
"""

# %% Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import linear_model

# %% Helper Functions
def lin_reg(X, y):
    """
    Function to execute the linear regression model
    """

    b_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    return b_hat


def lin_reg_ridge(X, y, lamb):
    """
    Function to execute the ridge regression model
    """

    b_hat = np.linalg.inv(X.T @ X + lamb * np.identity(X.shape[1])) @ X.T @ y
    b_o = np.mean(y)

    return b_hat, b_o


def mean_squared_error(y, y_hat):
    """
    Function for calculating the mean squared error of a given vector
    """
    mse = np.mean(np.square(y-y_hat))
    
    return mse

def standard_error(X, y, y_hat):
    """
    Get standard error for regular linear regression (used for z-score and table)
    """

    stde = 1/2 * np.sqrt((1 / (X.shape[0] - X.shape[1] - 1 ) * np.sum(np.square(y-y_hat))) * np.diagonal(np.linalg.inv(X.T @ X)))
    col_stde = np.array([stde]).T

    return col_stde


def test_lin_reg_ridge(X, y, b_hat, b_o):
    """
    Function to test the Ridge regression
    """
    y_hat = b_o + X @ b_hat
    
    return mean_squared_error(y, y_hat)


def test_lin_reg(X, y, b_hat):
    """
    Function to test linear regression
    """
    y_hat = X @ b_hat
    
    return mean_squared_error(y, y_hat), standard_error(X, y, y_hat)


def table_ridge_lasso(b_hat, tab, cols, skip=0):
    """
    Function to showcase data in tabular form
    """
    for i, item in enumerate(cols):
        tab.append((item, b_hat[i+skip]))
    
    return tab


def table_lin_reg(b_hat, standard_error, z_score, cols):
    """
    Function to showcase data in tabular form
    """
    tab = [("Term", "Coefficient", "Std. Error", "Z Score")]
    for i, item in enumerate(cols):
        B = np.round(b_hat[i], 3).squeeze()
        stde = np.round(standard_error[i], 3).squeeze()
        zscore = np.round(z_score[i], 3).squeeze()
        tab.append((item, B, stde, zscore))
    
    return tab



# %% Apply Regression Models

def apply_models(df, output):
    """
    Clean data and apply three distinct linear regression models

    1. Plain Old Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    """

    # ---------Data Cleaning-----------
    test_validation = df[df["train"] == "F"].drop("train", axis=1)
    training = df[df["train"] == "T"].drop("train", axis=1)

    # Normalized data 
    training = (training - training.mean()) / training.std()
    test_validation = (test_validation - test_validation.mean()) / test_validation.std()

    # Save the output vectors for training / testing (lpsa column)
    output_training = output[output["train"] == "T"].drop("train", axis=1).to_numpy()
    output_test = output[output["train"] == "F"].drop("train", axis=1).to_numpy()
    # output = (output - training.mean()) / training.std()

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

    # ---------Linear Regression---------
    b_hat = lin_reg(training_matrix, output_training)
    linear_mse, linear_stderr = test_lin_reg(test_matrix, output_test, b_hat)
    z_score = b_hat / linear_stderr
    
    cols = training.columns.values
    np.insert(cols, 0, "Intercept")
    tab = table_lin_reg(b_hat, linear_stderr, z_score, cols)
    linear_mse_train, linear_stderr_train = test_lin_reg(training_matrix, output_training, b_hat)
    
    corr = training.corr()

    print(f"Table 3.1: \n{tabulate(tab)} \n")
    print(f"Table 3.2: \n {corr} \n")
    print(f"Linear Regression MSE: {linear_mse} \n")
    print(f"Linear Regression MSE (Training): {linear_mse_train} \n")
    

    # --------Ridge Regression----------
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
    tab = table_ridge_lasso(b_ridge_best, tab, cols=training.columns)
    print(f"Best Beta_Ridge:\n {tabulate(tab)} \n")
    ridge_mse = test_lin_reg_ridge(test_matrix_split[:, 1:], output_test_split, b_ridge_best, b_o_best)
    ridge_mse_train = test_lin_reg_ridge(training_matrix[:, 1:], output_training, b_ridge_best, b_o_best)
    print(f"Ridge MSE: {ridge_mse} \n")
    print(f"Ridge MSE (Training): {ridge_mse_train} \n")

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.log(lambdas), betas.T)
    ax.legend(training.columns)
    ax.set_xlabel("Log(Lambda)")
    ax.set_ylabel("Coefficient")
    ax.invert_xaxis()
    ax.axvline(np.log(lamb_best), ls="--")
    ax.set_title("Ridge Regression Coefficients")
    plt.show()

    # ------------Lasso Regression-------------
    alpha = np.linspace(0.001, 1, 1000)
    best_score = float("-inf")
    best_parameters = {}
    betas = np.zeros((training_matrix.shape[1] - 1, 1000))
    for i, alpha2 in enumerate(alpha):
        clf = linear_model.Lasso(alpha=alpha2)
        clf.fit(training_matrix[:, 1:].tolist(), output_training.tolist())
        score = clf.score(validation_matrix[:, 1:], validation_output)
        # score = test_lin_reg_ridge(validation_matrix[:, 1:], validation_output, clf.coef_, clf.intercept_)
        betas[:, i] = clf.coef_
        if score > best_score:
            best_score = score
            best_alpha = alpha2
            best_parameters["weights"] = clf.coef_
            best_parameters["intercept"] = clf.intercept_    

    tab = []
    tab.append(("Intercept", best_parameters["intercept"]))
    tab = table_ridge_lasso(best_parameters["weights"], tab, cols=training.columns)
    print(f"Best Beta_Lasso:\n {tabulate(tab)} \n")


    lasso_mse = test_lin_reg_ridge(test_matrix_split[:, 1:], output_test_split, best_parameters["weights"], best_parameters["intercept"])
    lasso_mse_train = test_lin_reg_ridge(training_matrix[:, 1:], output_training, best_parameters["weights"], best_parameters["intercept"])
    print(f"Lasso MSE: {lasso_mse} \n")
    print(f"Lasso MSE (Training): {lasso_mse_train} \n")

    fig, ax = plt.subplots(1, 1)
    ax.plot(alpha, betas.T)
    ax.legend(training.columns)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Coefficient")
    ax.invert_xaxis()
    ax.axvline(best_alpha, ls="--")
    ax.set_title("Lasso Regression Coefficients")
    plt.show()


def main():
    #  Apply first Prostate Cancer Dataset
    df = pd.read_csv("prostate.txt", sep="\t")

    output = df[["lpsa", "train"]]
    df = df.drop(["Unnamed: 0", "lpsa"], axis=1)

    # Adding nonlinear term (after data exploration)
    df["logPGG"] = np.exp(df["pgg45"])
    df.drop("pgg45", axis=1, inplace=True)
    
    # Adding interaction term (again after data analysis)
    df["lcavol * age"] = - df["lcavol"] * df["age"]
    
    apply_models(df, output)

    #  Repeat for Real Estate dataset
    df2 = pd.read_csv("Real estate.csv")

    df2 = df2.drop(["No"], axis=1)
    df2['train'] = "T"
    dfupdate=df2.sample(50)
    dfupdate.train = "F"
    df2.update(dfupdate)

    output2 = df2[["Y house price of unit area", "train"]]
    df2.drop(["Y house price of unit area"], axis=1, inplace=True)

    apply_models(df2, output2)


if __name__ == "__main__":
    main()
