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
SUPPORTED_MODELS = ("random_forest", "logistic_regression", "lightgbm")


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

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            n_estimators=int(model_params["n_estimators"]),
            max_depth=int(model_params["max_depth"]),
            learning_rate=float(model_params.get("learning_rate", 0.1)),
            random_state=int(model_params["random_state"]),
            class_weight=class_weight,
            n_jobs=-1,
            verbosity=-1,
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
    title: str = "Matriz de confusion",
) -> None:
    """Save a confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    labels = sorted(pd.unique(pd.concat([y_true, pd.Series(predictions)], ignore_index=True)))
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    path = ensure_parent(output_path)

    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
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


def train_model(params: dict, *, write_dvc_outputs: bool = False) -> dict[str, float]:
    """Train one classifier and log the run to MLflow."""
    import joblib
    import mlflow
    import mlflow.sklearn

    train_data = pd.read_csv(params["data"]["processed_train_path"])
    test_data = pd.read_csv(params["data"]["processed_test_path"])

    feature_columns = get_feature_columns(train_data)
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET_COLUMN]
    X_test = test_data[feature_columns]
    y_test = test_data[TARGET_COLUMN]

    model_type = params["model"]["type"]
    pipeline = build_model(params["model"])
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=model_type):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        model_path = ensure_parent(f"models/{model_type}_model.joblib")
        metrics_path = ensure_parent(f"reports/{model_type}_metrics.json")
        predictions_path = ensure_parent(f"reports/{model_type}_predictions.csv")
        report_path = ensure_parent(f"reports/{model_type}_classification_report.txt")
        plot_path = Path(f"reports/{model_type}_confusion_matrix.png")

        joblib.dump(pipeline, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(report, encoding="utf-8")
        pd.DataFrame({"actual": y_test, "prediction": predictions}).to_csv(
            predictions_path, index=False
        )
        save_confusion_matrix_plot(
            y_test,
            predictions,
            str(plot_path),
            title=f"Matriz de confusion - {model_type}",
        )

        if write_dvc_outputs:
            ensure_parent("models/classification_model.joblib")
            joblib.dump(pipeline, "models/classification_model.joblib")
            Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            pd.DataFrame({"actual": y_test, "prediction": predictions}).to_csv(
                "reports/predictions.csv", index=False
            )
            save_confusion_matrix_plot(y_test, predictions, "reports/confusion_matrix.png")
            Path("reports/classification_report.txt").write_text(report, encoding="utf-8")

        mlflow.log_params(
            {
                "model_type": model_type,
                "n_estimators": params["model"].get("n_estimators"),
                "max_depth": params["model"].get("max_depth"),
                "max_iter": params["model"].get("max_iter"),
                "learning_rate": params["model"].get("learning_rate"),
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

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    metrics = train_model(params, write_dvc_outputs=True)
    print(f"Metricas guardadas para {params['model']['type']}: {metrics}")


if __name__ == "__main__":
    main()
