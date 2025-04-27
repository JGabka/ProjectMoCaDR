import argparse, pickle
import pandas as pd
from pathlib import Path
from models.train_functions import *
from models.predict_functions import predict


def parse_args():
    p = argparse.ArgumentParser(description='Movie Recommender System')
    p.add_argument('--train_file', required=True, help='Training data CSV')
    p.add_argument('--test_file', required=True, help='Test data CSV')
    p.add_argument('--alg', choices=['NMF', 'SVD1', 'SVD2', 'SGD'], required=True,
                   help='Algorithm to run')
    p.add_argument('--model_path', required=True, help='Path to save or load the model pickle')
    p.add_argument('--output_file', required=True, help='Path to save predictions CSV')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Load train and test data
    train_path = args.train_file
    test_path = args.test_file

    # Ensure output directories exist
    Path(Path(args.model_path).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(args.output_file).parent).mkdir(parents=True, exist_ok=True)

    # Training
    models = {}
    to_run = [args.alg]
    for alg in to_run:
        train_fn = globals()[f'train_{alg.lower()}_model']
        models[alg] = train_fn(train_path)

    # Save models
    with open(args.model_path, 'wb') as f:
        pickle.dump(models, f)

    # Prediction
    with open(args.model_path, 'rb') as f:
        loaded_models = pickle.load(f)
    preds = predict(test_path, loaded_models[args.alg])
    pd.DataFrame(preds) \
        .rename(columns={'userId': 'UserId', 'movieId': 'MovieId', 'rating': 'Rating'}) \
        [['UserId', 'MovieId', 'Rating']] \
        .to_csv(args.output_file, index=False)
