# Python Built-Ins:
import argparse
import os

# External Dependencies:
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ---- INFERENCE FUNCTIONS ---- Not implemented by default for sklean containers
# Takes a model path and return an sklearn model
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


if __name__ == "__main__":
    # ---- TRAINING ENTRY POINT ----
    
    # Arguments like data location and hyper-parameters are passed from SageMaker to your script
    # via command line arguments and/or environment variables. You can use Python's built-in
    # argparse module to parse them:
    print("Parsing training arguments")
    parser = argparse.ArgumentParser()

    # RandomForest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=10)
    parser.add_argument("--min_samples_leaf", type=int, default=3)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--features", type=str)  # explicitly name which features to use
    parser.add_argument("--target_variable", type=str)  # name the column to be used as target

    args, _ = parser.parse_known_args()

    # -- DATA PREPARATION --
    # Load the data from the local folder(s) SageMaker pointed us to:
    print("Reading data")
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    print("Building training and testing datasets")
    X_train = train_df[args.features.split()]
    X_test = test_df[args.features.split()]
    y_train = train_df[args.target_variable]
    y_test = test_df[args.target_variable]

    # -- MODEL TRAINING --
    print("Training model")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1)

    model.fit(X_train, y_train)

    # -- MODEL EVALUATION --
    print("Testing model")
    predictions = model.predict(X_test)
    abs_err = np.abs(predictions - y_test)

    # Percentile absolute errors:
    for q in [10, 50, 90]:
        print(f"AE-at-{q}th-percentile: {np.percentile(a=abs_err, q=q)}")

    # Additional metrics:
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_err)
    r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2-score: {r2}")

    # -- SAVE THE MODEL --
    # ...To the specific folder SageMaker pointed us to:
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print(f"model saved at {path}")
