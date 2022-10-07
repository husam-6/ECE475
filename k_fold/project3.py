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


def main():
    
    # From textbook:
    # N = 50 samples, 2 equal sized classes
    # p = 5000 quantitative predictors
    train = np.random.normal(loc=0, scale=1, size=(50, 5000))

    labels = np.randint(2, size = (50, 1))
    tmp = np.concatenate((labels, train), axis=1)
    
    overall_df = pd.DataFrame(tmp)
    overall_df.rename({0: "Labels"}, axis=1, inplace=True)

    #Wrong Way
    predictors_wrong, labels_wrong = get_top_100_predictors(overall_df)

    
    #Right Way
    kf = KFold(n_splits=5, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(train):
        print("TRAIN:", train_index, "TEST:", test_index)
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]

        
    


    
    return

# %% 
train = np.random.normal(loc=0, scale=1, size=(50, 5000))

labels = np.random.randint(2, size = (50, 1))
tmp = np.concatenate((labels, train), axis=1)

overall_df = pd.DataFrame(tmp)
overall_df.rename({0: "Labels"}, axis=1, inplace=True)

def get_top_100_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Function to grab the top 100 correlated predictors of the given dataset"""

    assert isinstance(df, pd.DataFrame)

    correlations = df.corr()["Labels"]
    correlations

    top_100_features = correlations.nlargest(101).index.values[1:] - 1
    predictors = df[top_100_features]
    labels = df["Labels"]

    return predictors, labels

# %%



# %%

# Wrong way - get top 100 correlations. Training set is seeing validation set... 
predictors, labels = get_top_100_predictors(overall_df)

kf = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in kf.split(train):
    # K = 1 Nearest Neighbor Classifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    X_train, X_test = predictors[train_index], predictors[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    neigh.fit(X_train, y_train)



# %%

predictors



# %%
if __name__ == "__main__":
    main()