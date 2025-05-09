import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

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
        user_means = np.nanmean(Z, axis=1)
        movie_means = np.nanmean(Z, axis=0)

        user_counts = np.sum(~np.isnan(Z), axis=1)  # liczba ocen u użytkownika
        movie_counts = np.sum(~np.isnan(Z), axis=0)  # liczba ocen filmu

        inds = np.where(np.isnan(Z))
        for i, j in zip(inds[0], inds[1]):
            user_mean = user_means[i]
            movie_mean = movie_means[j]
            user_count = user_counts[i]
            movie_count = movie_counts[j]

            if np.isnan(user_mean) and np.isnan(movie_mean):
                Z[i, j] = 0.0
            elif np.isnan(user_mean):
                Z[i, j] = movie_mean
            elif np.isnan(movie_mean):
                Z[i, j] = user_mean
            else:
                # Wagi proporcjonalne do liczby ocen
                total = user_count + movie_count
                if total == 0:
                    Z[i, j] = 0.5 * user_mean + 0.5 * movie_mean  # fallback
                else:
                    user_weight = user_count / total
                    movie_weight = movie_count / total
                    Z[i, j] = user_weight * user_mean + movie_weight * movie_mean

    else:
        raise ValueError("fillna_method must be one of: 'zero', 'movie', 'user'")

    return Z, user_map, movie_map


def train_svd2_model(train_file, n_components, limit=20):
    Z_true, user_map, movie_map = build_rating_matrix(train_file,'zero')
    Z, user_map, movie_map = build_rating_matrix(train_file,'weighted')
    Z_approx = Z.copy()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    p = 0

    while p < limit:
        svd.fit(Z_approx)
        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z_approx) / (svd.singular_values_)
        H = np.dot(Sigma2, VT)
        Z_new = np.dot(W, H)

        Z_approx = np.where(Z_true != 0, Z_true, Z_new)

        p += 1

    return {"Z_approx":Z_approx, "user_map":user_map, "movie_map":movie_map}

model=train_svd2_model("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/ProjectMoCaDR/test1.csv")


def predict(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)

    Z_approx = model_data["Z_approx"]
    user_map = model_data["user_map"]
    movie_map = model_data["movie_map"]

    predictions = []
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        if u in user_map and m in movie_map:
            i = user_map[u]
            j = movie_map[m]
            rating = Z_approx[i, j]
        else:
            # If user or movie not seen in training, default to 0 (or any strategy)
            rating = 0
        # Round rating to nearest 0.5 increment
        rating_rounded = round(rating * 2) / 2

        predictions.append({
            "userId": u,
            "movieId": m,
            "rating": rating_rounded
        })
    return predictions

prediction=predict("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/ProjectMoCaDR/sample_test.csv",model)

