import pandas as pd
import numpy as np

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
    elif fillna_method == 'weighted':
        # Obliczamy średnie dla użytkowników i filmów
        user_means = np.nanmean(Z, axis=1)
        movie_means = np.nanmean(Z, axis=0)

        inds = np.where(np.isnan(Z))
        for i, j in zip(inds[0], inds[1]):
            # Średnia ważona: np. 0.5 * user_mean + 0.5 * movie_mean
            user_mean = user_means[i]
            movie_mean = movie_means[j]
            if np.isnan(user_mean) and np.isnan(movie_mean):
                Z[i, j] = 0.0  # fallback gdy oba są nan
            elif np.isnan(user_mean):
                Z[i, j] = movie_mean
            elif np.isnan(movie_mean):
                Z[i, j] = user_mean
            else:
                Z[i, j] = 0.5 * user_mean + 0.5 * movie_mean

    else:
        raise ValueError("fillna_method must be one of: 'zero', 'movie', 'user'")

    return Z, user_map, movie_map