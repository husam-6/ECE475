"""Michael Bentivegna and Husam Almanakly

Frequentist ML Project 3
"""

# %% Libraries
import pandas as pd
import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, InitVar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def get_top_100_predictors(df: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    """
    Function to grab the top 100 correlated predictors of the given dataset
    """

    assert isinstance(df, pd.DataFrame)

    
    # Correlation values
    correlations = df.iloc[index, :].corr()["Labels"]
        
    top_100_features = correlations.nlargest(101).index.values[1:]
        
    
    # Grab predictors with highest correlation
    predictors = df[top_100_features]
    labels = df["Labels"]

    return predictors, labels


def main():
    # From textbook: N = 50 samples, 2 equal sized classes
    # p = 5000 quantitative predictors
    p = 5000
    N = 50
    num_category = int(N / 2)

    # Generate data
    train = np.random.normal(loc=0, scale=1, size=(N, p))
    labels = np.concatenate([np.ones(num_category), np.zeros(num_category)])
    labels = np.resize(labels, (50,1))
    np.random.shuffle(labels)

    # Throw into dataframe...
    tmp = np.concatenate((labels, train), axis=1)
    overall_df = pd.DataFrame(tmp)
    overall_df.rename({0: "Labels"}, axis=1, inplace=True)

    # Wrong way - get top 100 correlations. Training set is seeing validation set... 
    predictors, labels = get_top_100_predictors(overall_df, np.arange(0, 50, 1))

    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    total_acc = 0
    simulations = 50
    for _ in range(simulations):
        for train_index, test_index in kf.split(train):
            # K = 1 Nearest Neighbor Classifier
            
            neigh = KNeighborsClassifier(n_neighbors=1)
            X_train, X_test = predictors.iloc[train_index, :], predictors.iloc[test_index, :]
            y_train, y_test = labels[train_index], labels[test_index]

            
            neigh.fit(X_train.to_numpy(), y_train.to_numpy())
            total_acc += neigh.score(X_test.to_numpy(), y_test.to_numpy())


    print(f"Wrong KFold Accuracy: {total_acc /  (kf.n_splits * simulations)}")


    # Right way - correlation inside the loop
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    total_acc = 0
    simulations = 10
    all_test = []
    for _ in range(simulations):
        for train_index, test_index in kf.split(train):
            # Throw into dataframe...
            predictors2, labels2 = get_top_100_predictors(overall_df, train_index)
            all_test = test_index
            # K = 1 Nearest Neighbor Classifier
            neigh = KNeighborsClassifier(n_neighbors=1)
            X_train, X_test = predictors2.iloc[train_index, :], predictors2.iloc[test_index, :]
            y_train, y_test = labels2[train_index], labels2[test_index]
                    
            neigh.fit(X_train.to_numpy(), y_train.to_numpy())
            total_acc += neigh.score(X_test.to_numpy(), y_test.to_numpy())

    print(f"Correct KFold Accuracy: {total_acc /  (kf.n_splits * simulations)}")


    predictors.insert(0, "labels", labels)
    predictors.corr()["labels"][1:].hist()
    plt.title("Wrong Way")
    plt.show()

    
    predictors2.insert(0, "labels", labels[test_index])
    predictors2.iloc[all_test, :].corr()["labels"][1:].hist()
    plt.title("Right Way")
    plt.show()


if __name__ == "__main__":
    main()