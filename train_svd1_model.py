import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD



def train_svd1_model(train_file, n_components=5):
    Z, user_map, movie_map = build_rating_matrix(train_file)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(Z)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_

    W = svd.transform(Z) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    Z_approx = np.dot(W, H)


    return {"Z_approx":Z_approx, "user_map":user_map, "movie_map":movie_map}