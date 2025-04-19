import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def build_rating_matrix(train_file):
    """
    Reads a ratings CSV file with columns: userId, movieId, rating.
    Builds and returns the user–movie matrix Z (missing entries set to 0),
    along with mappings from userId to row index and movieId to column index.

    Parameters:
      - train_file (str): Path to the training CSV file.

    Returns:
      - Z (ndarray): Rating matrix of shape (n_users, n_movies).
      - user_map (dict): Mapping from userId to row index.
      - movie_map (dict): Mapping from movieId to column index.
    """
    df = pd.read_csv(train_file)

    # Extract unique users and movies
    unique_users = df["userId"].unique()
    unique_movies = df["movieId"].unique()

    # Create mappings: userId -> row index, movieId -> column index
    user_map = {uid: i for i, uid in enumerate(sorted(unique_users))}
    movie_map = {mid: j for j, mid in enumerate(sorted(unique_movies))}

    n_users = len(user_map)
    n_movies = len(movie_map)

    # Build matrix Z with zeros for missing ratings
    Z = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in df.itertuples():
        u = row.userId
        m = row.movieId
        rating = row.rating
        i = user_map[u]
        j = movie_map[m]
        Z[i, j] = rating

    return Z, user_map, movie_map


def train_svd2_model(train_file, n_components=5, limit=10):
    Z, user_map, movie_map = build_rating_matrix(train_file)
    Z_approx = Z.copy()
    p = 0

    while p < limit:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(Z_approx)

        Sigma2 = np.diag(svd.singular_values_)
        VT = svd.components_
        W = svd.transform(Z_approx) / (svd.singular_values_ + 1e-10)  # zabezpieczenie
        H = np.dot(Sigma2, VT)
        Z_new = np.dot(W, H)

        Z_approx = np.where(Z != 0, Z, Z_new)

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

