import pandas as pd


def predict(test_file, model_data):
    """
    Reads a test CSV with columns: userId, movieId.
    Uses the stored Z_approx, user_map, movie_map to produce predictions.
    Missing userId/movieId combos produce a default rating (e.g., 0 or average).

    Returns a list of dicts with keys: 'userId', 'movieId', 'rating'.
    """
    df = pd.read_csv(test_file)
    z_approx = model_data['Z_approx']
    umap, mmap = model_data['user_map'], model_data['movie_map']
    predictions = []
    for u, m in df[['userId', 'movieId']].itertuples(index=False):
        if u in umap and m in mmap:
            r = z_approx[umap[u], mmap[m]]
        else:
            r = 0

        predictions.append({'userId': u,
                            'movieId': m,
                            'rating': round(r*2)/2})
    return predictions
