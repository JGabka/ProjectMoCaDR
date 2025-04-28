import pandas as pd

def calculate_rmse(test_file, predictions):
    df_true = pd.read_csv(test_file)
    df_pred = pd.DataFrame(predictions)
    merged = pd.merge(df_true, df_pred, on=["userId", "movieId"], suffixes=("_true", "_pred"))

    y_true = merged["rating_true"].values
    y_pred = merged["rating_pred"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def calculate_rmse_sgd(train_file, Z_approx):
  Z, user_map, movie_map = build_rating_matrix(train_file,'zero')
  I=set(zip(*np.nonzero(Z)))
  train_set, test_set = split_indices(I, seed=5)
  rows = torch.tensor([i[0] for i in test_set], device=device)
  cols = torch.tensor([i[1] for i in test_set], device=device)
  Z = torch.from_numpy(Z)
  loss = torch.mean((Z[rows, cols] - Z_approx[rows, cols]) ** 2)
  rmse=loss**(1/2)
  return rmse