"""
Michael Bentivegna and Husam Almanakly
Frequentist ML Project 5 Stretch Goal
"""

# %% Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.001
# %%

def calc_loss(P, U, df, user_map, movie_map):
    """ Calculate loss for NMF algorithm"""
    total = 0
    for i, row in df.iterrows():
        i = user_map[int(row["userId"])]
        j = movie_map[int(row["movieId"])]

        # Equation 2 in notes
        total += (row["rating"] - U[i, :] @ P[j, :]) ** 2 + LAMBDA * (np.linalg.norm(U[i, :]) + np.linalg.norm(P[j, :]) )

    return total


def create_initial_noisy_matrix(m, n):
    """ Create m by n matrix with gaussian noise"""

    return np.random.uniform(0, 1, (m, n))


def update_matrix(X, y):
    """
    X is a (latent dimension by # of items / user matrix)
    lam is the regularization constant
    y is the ith row of the ratings matrix

    Update one row in the user / item 
    """
    
    return np.linalg.inv(X.T @ X + LAMBDA * np.identity(X.shape[1])) @ X.T @ y

def get_by_id(df: pd.DataFrame, id: int, id_type="userId") -> pd.DataFrame:
    """ Get all relevant rows in data"""
    return df[df[id_type] == id]


# Loop through dataset
def get_r_for_p_update(movie_df: pd.DataFrame, num_users, user_map) -> np.ndarray:
    """ Get padded r vector needed for item update equation"""
    
    r_vec = np.zeros((num_users, 1))
    
    # Loop through and insert into row vector
    for k, row in movie_df.iterrows():
        i = int(row["userId"])
        i = user_map[i]
        r_vec[i] = row["rating"]
    
    return r_vec


def get_r_for_u_update(user_df: pd.DataFrame, num_movies, movie_map) -> np.ndarray:
    """ Get padded r vector needed for user update equation"""
    r_vec = np.zeros((num_movies, 1))
    
    # Loop through and insert into row vector
    for k, row in user_df.iterrows():
        i = int(row["movieId"])
        i = movie_map[i]
        r_vec[i] = row["rating"]
    
    return r_vec


def main():
    # Read in Dataset
    df = pd.read_csv("ml-25m/ratings.csv").drop("timestamp", axis = 1)
    latent_var  = 100
    

    # Take small subset 
    df = df[df["userId"] < 200]
    df = df[df["movieId"] < 100]

    # Read in ratings data and convert to matrix
    user_ids = df["userId"].unique()
    movie_ids = df["movieId"].unique()
    user_ids.sort()
    movie_ids.sort()

    # Store the number of users and movies in our data...
    num_users = user_ids.shape[0]
    num_movies = movie_ids.shape[0]

    # Set up hash map to map Movie IDs to true numerical index 
    movie_map = {}
    for i in range(num_movies):
        curr = movie_ids[i]
        if curr not in movie_map:
            movie_map[curr] = i

    user_map = {}
    for i in range(num_users):
        curr = user_ids[i]
        if curr not in user_map:
            user_map[curr] = i

    # Initialize random matrices for P and Q
    U = create_initial_noisy_matrix(num_users, latent_var)
    P = create_initial_noisy_matrix(num_movies, latent_var)

    # Train using alternating least squares algorithm    
    loss = np.zeros(df.shape[0])
    j = 0
    for i, row in df.iterrows():
        # Hold P constant, update U
        user_ratings = get_by_id(df, row["userId"], "userId")
        r_u = get_r_for_u_update(user_ratings, num_movies, movie_map)
        U_row_updated = update_matrix(P, r_u)
        k = int(row["userId"])
        k = user_map[k]
        U[k, :] = U_row_updated.flatten()

        # Hold U constant, update P
        movie_ratings = get_by_id(df, row["movieId"], "movieId")
        r_p = get_r_for_p_update(movie_ratings, num_users, user_map)
        P_row_updated = update_matrix(U, r_p)
        k = movie_map[int(row["movieId"])]
        P[k, :] = P_row_updated.flatten()

        print(j)
        loss[j] = calc_loss(P, U, df, user_map, movie_map)
        j += 1

    np.save("subset", loss)
    plt.figure()
    plt.plot(loss)
    plt.show()


if __name__ == "__main__":
    main()