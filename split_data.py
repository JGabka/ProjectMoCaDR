import sys
import pyspark
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from recommenders.datasets.python_splitters import python_stratified_split

COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_PREDICTION = "Rating"
COL_TIMESTAMP = "Timestamp"


data = pd.read_csv("/content/ratings.csv", sep=",", names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP])

data_train, data_test = python_stratified_split(
    data, filter_by="user", min_rating=20, ratio=0.9,
    col_user=COL_USER, col_item=COL_ITEM)

data_train = data_train.rename(columns={"UserId": "userId", "MovieId": "movieId", "Rating": "rating"})
data_train[["userId", "movieId", "rating"]].to_csv("/content/data_train.csv", index=False)

data_test = data_test.rename(columns={"UserId": "userId", "MovieId": "movieId", "Rating": "rating"})
data_test[["userId", "movieId", "rating"]].to_csv("/content/data_test.csv", index=False)


