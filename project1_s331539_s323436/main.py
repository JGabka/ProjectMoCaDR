import argparse, pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.train_functions import *
from models.predict_functions import predict


def parse_args():
    p = argparse.ArgumentParser(description='Movie Recommender System')
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--data_file', help='Full ratings CSV to split')
    grp.add_argument('--train_file', help='Training data CSV')
    p.add_argument('--test_file', help='Test data CSV (if --train_file used)')
    p.add_argument('--split_ratio', type=float, default=None,
                   help='Fraction for test split (0< ratio <1)')
    p.add_argument('--alg', choices=['NMF', 'SVD1', 'SVD2', 'SGD', 'ALL'], required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--output_file', required=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Prepare train/test files with automatic naming
    if args.data_file:
        df = pd.read_csv(args.data_file)
        ratio = args.split_ratio or 0.2
        tr, te = train_test_split(df, test_size=ratio, random_state=42)
        stem = Path(args.data_file).stem
        train_path = f"data/{stem}_train.csv"
        test_path = f"data/{stem}_test.csv"
        tr.to_csv(train_path, index=False)
        te.to_csv(test_path, index=False)
    else:
        train_path, test_path = args.train_file, args.test_file

    # Ensure output dirs exist
    Path(Path(args.model_path).parent).mkdir(parents=True, exist_ok=True)
    Path(Path(args.output_file).parent).mkdir(parents=True, exist_ok=True)

    # Training
    models = {}
    to_run = [args.alg] if args.alg != 'ALL' else ['NMF', 'SVD1', 'SVD2', 'SGD']
    for alg in to_run:
        fn = globals()[f'train_{alg.lower()}_model']
        models[alg] = fn(train_path)
    with open(args.model_path, 'wb') as f:
        pickle.dump(models, f)

    # Prediction
    with open(args.model_path, 'rb') as f:
        loaded = pickle.load(f)
    preds = predict(test_path, loaded[args.alg])
    pd.DataFrame(preds).to_csv(args.output_file, index=False)
