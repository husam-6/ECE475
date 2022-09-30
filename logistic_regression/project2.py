# %% Libraries
import pandas as pd
import numpy as np
import os
import tqdm

# %% data processing

# Read in data - remove labels and divide into training / validation / test
script_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(f"{script_path}/SAheart.csv").drop("row.names",axis=1)
data.insert(0, "intercept", np.ones(data.shape[0]))
labels = data["chd"].to_numpy()
data = data.drop("chd", axis=1)

# Convert Famhist to 0's and 1's
tmp = data["famhist"].values
tmp = (tmp == "Present") * 1
data["famhist"] = tmp

feature_names = data.columns.values     # Feature names in order
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



# %%

def cross_validation(y, y_hat):
    """
    y_hat is predicted probabilities of output values between 0 and 1
    y is the actual value with discrete values 0 or 1"""
    """"""
    
    return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def h_theta(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Function for h_theta(x)

    Pass in row of x's (1 x (p+1), features for 1 sample) and 
    row vector of weights (1 x (p+1) thetas)
    """
    return 1 / (1 + np.exp(- theta * x.T))


def update_weights_without_l2(x, y, learning_rate, old_theta):
    """
    x is 1 x (p+1) for one sample 
    y is a scalar 0 or 1
    learning rate is scalar
    theta is 1 x (p+1) of weights
    """
    
    return old_theta + learning_rate * (y - h_theta(x, old_theta)) * x


def sgd(iterations, x, y, learning_rate, initial_theta):
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
        
        theta = update_weights_without_l2(x[chosen, :], y[chosen, 0], learning_rate, theta)

        y_hat = h_theta(x, theta)
        loss = cross_validation(y, y_hat)
        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    return theta

# %%
def main():
    
    #normalize starting theta values
    theta = np.random.normal(size=(1, x.shape[1]))
    
    # Stochastic gradient descent call
    output = sgd(10000, x, y, .1, theta)
    
    return 0


if __name__ == "__main__":
    main()