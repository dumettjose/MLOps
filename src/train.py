import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ensure_parent, load_params


TARGET_COLUMN = "engagement_level"


def get_feature_columns(data: pd.DataFrame) -> list[str]:
    """Return feature columns from preprocessed train/test CSVs."""
    return [column for column in data.columns if column != TARGET_COLUMN]


def build_model(model_params: dict) -> Pipeline:
    """Build a classification pipeline from params.yaml."""
    model_type = model_params["type"]
    class_weight = model_params.get("class_weight")

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=int(model_params.get("max_iter", 1000)),
            random_state=int(model_params["random_state"]),
            class_weight=class_weight,
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(model_params["n_estimators"]),
            max_depth=int(model_params["max_depth"]),
            random_state=int(model_params["random_state"]),
            class_weight=class_weight,
            n_jobs=-1,
        )
        return Pipeline(steps=[("classifier", classifier)])

    raise ValueError(f"Modelo no soportado: {model_type}")


def evaluate(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Compute standard multiclass classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }


def save_confusion_matrix_plot(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_path: str,
) -> None:
    """Save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    labels = sorted(pd.unique(pd.concat([y_true, pd.Series(predictions)], ignore_index=True)))
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    path = ensure_parent(output_path)

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de confusion")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            plt.text(col_index, row_index, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    import joblib
    import mlflow
    import mlflow.sklearn

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    train_data = pd.read_csv(params["data"]["processed_train_path"])
    test_data = pd.read_csv(params["data"]["processed_test_path"])

    feature_columns = get_feature_columns(train_data)
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[feature_columns]
    y_test = test_data[TARGET_COLUMN]

    pipeline = build_model(params["model"])
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=params["model"]["type"]):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        model_path = ensure_parent("models/classification_model.joblib")
        metrics_path = ensure_parent("reports/metrics.json")
        predictions_path = ensure_parent("reports/predictions.csv")
        report_path = ensure_parent("reports/classification_report.txt")
        plot_path = Path("reports/confusion_matrix.png")

        joblib.dump(pipeline, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(report, encoding="utf-8")
        pd.DataFrame({"actual": y_test, "prediction": predictions}).to_csv(
            predictions_path, index=False
        )
        save_confusion_matrix_plot(y_test, predictions, str(plot_path))

        mlflow.log_params(
            {
                "model_type": params["model"]["type"],
                "n_estimators": params["model"].get("n_estimators"),
                "max_depth": params["model"].get("max_depth"),
                "class_weight": params["model"].get("class_weight"),
                "model_random_state": params["model"].get("random_state"),
                "data_random_state": params["data"]["random_state"],
                "n_samples": params["data"]["n_samples"],
                "test_size": params["split"]["test_size"],
                "stratify": params["split"].get("stratify"),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(predictions_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    print(f"Metricas guardadas en {metrics_path}: {metrics}")


if __name__ == "__main__":
    main()
