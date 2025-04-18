import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dist4(Z, WH):
    return torch.mean(torch.pow(Z - WH, 4))


lr = 0.02
n_epochs = 2000

def train_sgd_model(train_file, lr=0.02, n_epochs=20):

    Z, user_map, movie_map = build_rating_matrix(train_file)
    n,d = Z.shape[0], Z.shape[1]
    Z = torch.from_numpy(Z)
    for r in range(20):
        W = torch.randn((n,r), requires_grad=True, dtype=torch.float, device=device)
        H = torch.randn((r,d), requires_grad=True, dtype=torch.float, device=device)
        optimizer = torch.optim.Adam([W, H], lr=lr)
        loss_list = []

        for epoch in range(n_epochs):

            loss=torch.mean(torch.pow(Z-torch.matmul(W,H),2))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        print("Distance using SGD with r=",r,loss)



model = train_sgd_model("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/sample_project1-2/tools/ratings.csv")

# plt.figure(figsize=(12,4))
# plt.plot(loss_list, 'r')
# plt.grid('True', color='y')
# plt.show()



matrix,user, movie = build_rating_matrix("/Users/juliagabka/Desktop/studia/magisterka /1 rok/2 semestr/mocadr/sample_project1-2/tools/ratings.csv")
print(matrix.shape)