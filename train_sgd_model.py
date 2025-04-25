import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
np.random.seed(42)

def build_rating_matrix(train_file, fillna_method='zero'):
    """
    Reads a ratings CSV file with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z (missing entries set to 0 or mean),
    along with mappings from userId to row index and movieId to column index.

    Parameters:
      - train_file (str): Path to the training CSV file.

    Returns:
      - Z (ndarray): Rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    df = pd.read_csv(train_file)

    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    Z = np.full((n_users, n_movies), np.nan, dtype=np.float32)

    for row in df.itertuples():
        i = user_map[row.userId]
        j = movie_map[row.movieId]
        Z[i, j] = row.rating

    if fillna_method == 'movie':
        movie_means = np.nanmean(Z, axis=0)
        inds = np.where(np.isnan(Z))
        Z[inds] = np.take(movie_means, inds[1])

    elif fillna_method == 'user':
        user_means = np.nanmean(Z, axis=1)
        inds = np.where(np.isnan(Z))
        Z[inds] = np.take(user_means, inds[0])

    elif fillna_method == 'zero':
        Z = np.nan_to_num(Z, nan=0.0)

    else:
        raise ValueError("fillna_method must be one of: 'zero', 'movie', 'user'")

    return Z, user_map, movie_map

# def build_rating_matrix(train_file):
#     """
#     Reads a ratings CSV file with columns: userId, movieId, rating.
#     Builds and returns the user–movie matrix Z (missing entries set to 0),
#     along with mappings from userId to row index and movieId to column index.
#
#     Parameters:
#       - train_file (str): Path to the training CSV file.
#
#     Returns:
#       - Z (ndarray): Rating matrix of shape (n_users, n_movies).
#       - user_map (dict): Mapping from userId to row index.
#       - movie_map (dict): Mapping from movieId to column index.
#     """
#     df = pd.read_csv(train_file)
#
#     # Extract unique users and movies
#     unique_users = df["userId"].unique()
#     unique_movies = df["movieId"].unique()
#
#     # Create mappings: userId -> row index, movieId -> column index
#     user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
#     movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}
#
#     n_users = len(user_map)
#     n_movies = len(movie_map)
#
#     # Build matrix Z with zeros for missing ratings
#     Z = np.zeros((n_users, n_movies), dtype=np.float32)
#     for row in df.itertuples():
#         u = row.userId
#         m = row.movieId
#         rating = row.rating
#         i = user_map[u]
#         j = movie_map[m]
#         Z[i, j] = rating
#
#     return Z, user_map, movie_map
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_indices(indices, train_ratio=0.8, seed=None):
    indices_list = list(indices)
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices_list)

    split_point = int(len(indices_list) * train_ratio)
    train_set = set(indices_list[:split_point])
    test_set = set(indices_list[split_point:])

    return train_set, test_set

def train_sgd_model(train_file, lr=0.02, n_epochs=20):

    Z, user_map, movie_map = build_rating_matrix(train_file,'zero')
    n,d = Z.shape[0], Z.shape[1]
    I=set(zip(*np.nonzero(Z)))
    train_set, test_set = split_indices(I)
    Z = torch.from_numpy(Z)

    for r in range(1,20):
        W = torch.randn((n,r), requires_grad=True, dtype=torch.float, device=device)
        H = torch.randn((r,d), requires_grad=True, dtype=torch.float, device=device)
        optimizer = torch.optim.SGD([W, H], lr=lr)
        loss_list = []

        for epoch in range(n_epochs):
            epoch_loss_list = []
            pred = torch.matmul(W, H)
            rows = torch.tensor([i[0] for i in train_set], device=device)
            cols = torch.tensor([i[1] for i in train_set], device=device)

            loss = torch.mean((Z[rows, cols] - pred[rows, cols]) ** 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        print("Distance using SGD with r=",r,loss)
        # plt.figure(figsize=(12, 4))
        # plt.plot(loss_list, 'r')
        # plt.grid('True', color='y')
        # plt.show()



loss_list = train_sgd_model("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/ProjectMoCaDR/ratings.csv")





matrix,user, movie = build_rating_matrix("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/ProjectMoCaDR/test1.csv")
print(matrix.shape)
