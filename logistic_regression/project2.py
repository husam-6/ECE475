# %% Libraries
import pandas as pd
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, InitVar

# For SAHeart Data
@dataclass
class Data:
    # Training data
    training_data: np.ndarray = field(init=False)
    training_labels: np.ndarray = field(init=False)

    # Validation
    validation_data: np.ndarray = field(init=False)
    validation_labels: np.ndarray = field(init=False)
    
    # Test
    test_data: np.ndarray = field(init=False)
    test_labels: np.ndarray = field(init=False)

    # Arrays for names and labels
    labels_df: np.ndarray = field(init=False)
    feature_names: list = field(init=False)

    df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # Read in data - remove labels and divide into training / validation / test
        script_path = os.path.dirname(os.path.realpath(__file__))
        data = pd.read_csv(f"{script_path}/SAheart.csv").drop("row.names",axis=1)
        labels = data["chd"].to_numpy()
        data = data.drop(["adiposity", "typea", "chd"], axis=1)

        # Convert Famhist to 0's and 1's
        tmp = data["famhist"].values
        tmp = (tmp == "Present") * 1
        data["famhist"] = tmp
        self.feature_names = data.columns.values     # Feature names in order

        # Scale data
        data = (data - data.mean()) / data.std()
        data.insert(0, "intercept", np.ones(data.shape[0]))
        numpy_data = data.to_numpy()

        # Divide data into 80% training, 10% validation, 10% test
        num_samples = data.shape[0]
        training_cutoff = int(num_samples * 0.8)
        test_cutoff = training_cutoff + int(num_samples * 0.1)

        # Samples
        self.training_data = numpy_data[:training_cutoff, :]
        self.test_data = numpy_data[training_cutoff:test_cutoff, :]
        self.validation_data = numpy_data[test_cutoff:, :]

        # Labels
        self.training_labels = labels[:training_cutoff]
        self.test_labels = labels[training_cutoff:test_cutoff]
        self.validation_labels = labels[test_cutoff:]

        # Save dataframe as well
        self.df = data
        self.labels_df = labels


def cross_validation(y, y_hat):
    """
    y_hat is predicted probabilities of output values between 0 and 1
    y is the actual value with discrete values 0 or 1
    """
    
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
    x = x.reshape(-1, theta.shape[1])
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


def sgd(iterations, x, y, learning_rate, initial_theta, validation, l2_reg, display=True):
    """"
    x is N x (p+1)
    y is N x 1
    iterations and learning rate are user chosen scalars
    initial_theta is the starting weight values
    """
    theta = initial_theta
    
    if display:
        bar = tqdm.trange(iterations)
    else:
        bar = range(iterations)

    for i in bar:
        chosen = np.random.randint(y.shape[0])
        theta = update_weights(x[chosen, :], y[chosen], learning_rate, theta, l2_reg)

        y_hat = h_theta(x, theta)
        training_accuracy = calculate_accuracy(y, y_hat)
        training_prob = cross_validation(y, y_hat)

        y_val = h_theta(validation[0], theta)
        validation_accuracy = calculate_accuracy(validation[1], y_val)
        validation_prob = cross_validation(validation[1], y_val)
        
        if display:
            bar.set_description(
                f"prob => {training_prob:0.4f}, acc => {training_accuracy:0.4f}, val_prob => {validation_prob:0.4f}, val_acc => {validation_accuracy:0.4f}"
            )
            bar.refresh()

    return theta, validation_prob


def forward_stepwise(iterations, x, y, learning_rate, initial_theta, validation, test):
    """Function to implement stepwise analysis of weights
    """
    
    unchosen_indices = list(range(initial_theta.shape[1]))
    current_initial_thetas = np.array([]).reshape(1, 0)
    current_x = np.array([]).reshape(x.shape[0], 0)
    current_val = np.array([[]]).reshape(validation[0].shape[0], 0)
    current_test = np.array([[]]).reshape(test[0].shape[0], 0)

    # Choose best 5 parameters
    for j in range(5):
        best_prob = float("-inf")
        #best_index = int("-inf")
        for i in unchosen_indices:
            tmp_theta = np.concatenate((current_initial_thetas, initial_theta[0, i].reshape(1, -1)), axis = 1)
            tmp_current_x = np.concatenate((current_x, x[:, i].reshape(-1, 1)), axis = 1)
            tmp_current_val = np.concatenate((current_val, validation[0][:, i].reshape(-1, 1)), axis = 1)

            theta, prob = sgd(iterations, tmp_current_x, y, learning_rate,
                              tmp_theta, (tmp_current_val, validation[1]), l2_reg=False,
                              display=False
            )

            if best_prob < prob:
                best_prob = prob
                best_index = i

        
        unchosen_indices.remove(best_index)
        current_initial_thetas = np.concatenate((current_initial_thetas, initial_theta[0, best_index].reshape(1, -1)), axis = 1)
        current_x = np.concatenate((current_x, x[:, best_index].reshape(-1, 1)), axis = 1)
        current_val = np.concatenate((current_val, validation[0][:, best_index].reshape(-1, 1)), axis = 1)
        current_test = np.concatenate((current_test, test[0][:, best_index].reshape(-1, 1)), axis = 1)

    return current_initial_thetas, current_x, current_test


def plot_scatters(data):
    """Function to create big scatter plot figure for SA Dataset"""
    fig, ax = plt.subplots(7, 7, figsize=(12,12))
    class1 = data.df[data.labels_df == 0]
    class2 = data.df[data.labels_df == 1]

    # print(list(itertools.combinations(feature_names, 2)))
    for i in range(len(data.feature_names)):
        for j in range(len(data.feature_names)):
            if i == j:
                ax[i,i].text(0.2, 0.5, data.feature_names[i])
                continue
            class1.plot(kind='scatter', x=data.feature_names[j], y=data.feature_names[i], ax=ax[i,j],
                        facecolors='none', edgecolors="b", alpha=0.6)
            class2.plot(kind='scatter', x=data.feature_names[j], y=data.feature_names[i], ax=ax[i,j],
                        facecolors='none', edgecolors="r", alpha=0.6)
    
    fig.tight_layout()
    plt.show()


def print_loss(theta, updated_test, test_labels, output_string):
    y_hat = h_theta(updated_test, theta)
    loss = cross_validation(test_labels, y_hat)
    accuracy = calculate_accuracy(test_labels, y_hat)
    
    print("----------------" + output_string + "----------------" )
    print(f"Test Loss=> {loss:0.4f}, Test Accuracy => {accuracy:0.4f}")
    

def main():
    # Initialize data and weight vector
    data = Data()
    theta = np.zeros([1, data.training_data.shape[1]])
    
    # Stochastic gradient descent call without L2 regularization
    theta_out, val_prob = sgd(1000, data.training_data,
                           data.training_labels, 0.01, theta,
                           (data.validation_data, data.validation_labels),
                           l2_reg=False
    )
    print_loss(theta_out, data.test_data, data.test_labels, "SGD without L2")
    
    # Stochastic gradient descent call for L2 regularization
    theta_out_l2, val_prob = sgd(1000, data.training_data,
                           data.training_labels, 0.01, theta,
                           (data.validation_data, data.validation_labels),
                           l2_reg=True
    )
    print_loss(theta_out_l2, data.test_data, data.test_labels, "SGD with L2")

    # Stepwise analysis
    print("Applying stepwise analysis of theta vector...")
    theta_out_stepwise, updated_x, updated_test = forward_stepwise(1000, data.training_data, 
                     data.training_labels, 0.01, theta_out,
                     validation=(data.validation_data, data.validation_labels),
                     test=(data.test_data, data.test_labels)
    )

    print_loss(theta_out_stepwise, updated_test, data.test_labels, "SGD with Forward Step-Wise")

    plot_scatters(data)
    return


if __name__ == "__main__":
    main()