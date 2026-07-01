"""Exporta resumen de experimentos MLflow a evidence/ para versionar en Git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_params


def export_mlflow_evidence(params: dict, output_dir: str = "evidence") -> Path:
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No existe el experimento MLflow: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    exported_runs = []
    for run in runs:
        exported_runs.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
            }
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "mlflow_runs.json"
    summary_file.write_text(
        json.dumps(
            {
                "experiment_name": experiment_name,
                "tracking_uri": tracking_uri,
                "total_runs": len(exported_runs),
                "runs": exported_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output-dir", default="evidence")
    args = parser.parse_args()

    params = load_params(args.params)
    output_path = export_mlflow_evidence(params, output_dir=args.output_dir)
    print(f"Evidencia MLflow exportada en {output_path}")


if __name__ == "__main__":
    main()
