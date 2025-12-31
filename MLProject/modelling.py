import argparse
from pathlib import Path
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression


def load_processed(preproc_dir: Path):
    train_path = preproc_dir / "customer_churn_train.csv"
    test_path  = preproc_dir / "customer_churn_test.csv"
    target_path = preproc_dir / "target_name.txt"

    target_col = target_path.read_text(encoding="utf-8").strip()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test, target_col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc_dir", required=True)
    ap.add_argument("--tracking_uri", default="file:./mlruns")
    ap.add_argument("--experiment_name", default="CI_Retrain_Autolog")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X_train, y_train, X_test, y_test, target_col = load_processed(Path(args.preproc_dir))

    # BASIC: autolog
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="ci_retrain_logreg_autolog"):
        mlflow.set_tag("target", target_col)
        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=args.seed
        )
        model.fit(X_train, y_train)


if __name__ == "__main__":
    main()