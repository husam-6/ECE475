# %% Libraries
import pandas as pd
import numpy as np
import os
import tqdm

# %% data processing

# Read in data - remove labels and divide into training / validation / test
script_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(f"{script_path}/SAheart.csv").drop("row.names",axis=1)
labels = data["chd"].to_numpy()
data = data.drop("chd", axis=1)

# Convert Famhist to 0's and 1's
tmp = data["famhist"].values
tmp = (tmp == "Present") * 1
data["famhist"] = tmp

feature_names = data.columns.values     # Feature names in order
data = (data - data.mean()) / data.std()
data.insert(0, "intercept", np.ones(data.shape[0]))
numpy_data = data.to_numpy()

# Divide data - 80% training, 10% validation, 10% test
num_samples = data.shape[0]
training_cutoff = int(num_samples * 0.8)
test_cutoff = training_cutoff + int(num_samples * 0.1)

# Data
training_data = numpy_data[:training_cutoff, :]
test_data = numpy_data[training_cutoff:test_cutoff, :]
validation_data = numpy_data[test_cutoff:, :]

# Labels
training_labels = labels[:training_cutoff]
test_labels = labels[training_cutoff:test_cutoff]
validation_labels = labels[test_cutoff:]

# Normalize data
# training_data = (training_data - np.mean(training_data, axis=1)) / np.std(training_data, axis=1)
# training_data

# %%

def cross_validation(y, y_hat):
    """
    y_hat is predicted probabilities of output values between 0 and 1
    y is the actual value with discrete values 0 or 1"""
    """"""
    
    return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def calculate_accuracy(y, y_hat):
    guesses = (y_hat > 0.5) * 1
    return ((y == guesses) * 1).mean()


def h_theta(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Function for h_theta(x)

    Pass in row of x's (1 x (p+1), features for 1 sample) and 
    row vector of weights (1 x (p+1) thetas)
    """
    den = 1 + np.exp(- (theta @ x.T))
    return (1 / den).flatten()


def update_weights(x, y, learning_rate, theta, regularization=False):
    """
    x is 1 x (p+1) for one sample 
    y is a scalar 0 or 1
    learning rate is scalar
    theta is 1 x (p+1) of weights
    """ 
    # theta = theta + learning_rate * (y - h_theta(x, theta)) * x
    L2 = 0
    for j in range(theta.shape[1]):
        if regularization:
            L2 = 0.001 * theta[0, j]
        theta[0, j] = theta[0, j] + learning_rate * (y - h_theta(x, theta)) * x[j] - L2
    # j = np.random.randint(0, len(theta))
    # theta[j] = theta[j] + learning_rate * (y - h_theta(x, theta)) * x[j]

    return theta


def sgd(iterations, x, y, learning_rate, initial_theta, validation):
    """"
    x is N x (p+1)
    y is N x 1
    iterations and learning rate are user chosen scalars
    initial_theta is the starting weight values
    """
    theta = initial_theta
    bar = tqdm.trange(iterations)
    for i in bar:
        chosen = np.random.randint(y.shape[0])
        theta = update_weights(x[chosen, :], y[chosen], learning_rate, theta, True)

        y_hat = h_theta(x, theta)
        training_accuracy = calculate_accuracy(y, y_hat)
        training_prob = cross_validation(y, y_hat)
        
        y_val = h_theta(validation[0], theta)
        validation_accuracy = calculate_accuracy(validation[1], y_val)
        validation_prob = cross_validation(validation[1], y_val)
        
        bar.set_description(
            f"prob => {training_prob:0.4f}, acc => {training_accuracy:0.4f}, val_prob => {validation_prob:0.4f}, val_acc => {validation_accuracy:0.4f}"
        )
        bar.refresh()

    return theta

# %%
def main():
    
    #normalize starting theta values
    theta = np.random.normal(size=(1, training_data.shape[1]))
    
    # Stochastic gradient descent call
    output = sgd(10000, training_data, training_labels, 0.01, theta, (validation_data, validation_labels))

    y_hat = h_theta(test_data, output)
    loss = cross_validation(test_labels, y_hat)
    accuracy = calculate_accuracy(test_labels, y_hat)
    print(f"Test Loss => {loss:0.4f}, Test Accuracy => {accuracy:0.4f}")

    return


if __name__ == "__main__":
    main()