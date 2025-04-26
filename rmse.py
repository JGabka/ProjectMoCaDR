import pandas as pd

def calculate_rmse(test_file, predictions):
    df_true = pd.read_csv(test_file)
    df_pred = pd.DataFrame(predictions)
    merged = pd.merge(df_true, df_pred, on=["userId", "movieId"], suffixes=("_true", "_pred"))

    y_true = merged["rating_true"].values
    y_pred = merged["rating_pred"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

