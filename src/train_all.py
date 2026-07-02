import argparse
import copy
import json
import shutil
from pathlib import Path

from .config import load_params
from .export_mlflow_evidence import export_mlflow_evidence
from .train import SUPPORTED_MODELS, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y registra en MLflow los 3 modelos de clasificacion."
    )
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    comparison: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}

    for model_type in SUPPORTED_MODELS:
        run_params = copy.deepcopy(params)
        run_params["model"]["type"] = model_type
        print(f"\n=== Entrenando {model_type} ===")
        try:
            metrics = train_model(run_params)
            comparison[model_type] = metrics
            print(f"OK {model_type}: {metrics}")
        except OSError as error:
            message = (
                f"No se pudo entrenar {model_type}. En macOS instala libomp: brew install libomp"
            )
            errors[model_type] = message
            print(f"ERROR {model_type}: {message}\nDetalle: {error}")

    summary_path = Path("reports/model_comparison.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"metrics": comparison, "errors": errors}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResumen guardado en {summary_path}")

    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, evidence_dir / "model_comparison.json")

    if not comparison:
        raise SystemExit("Ningun modelo pudo entrenarse.")

    mlflow_evidence = export_mlflow_evidence(params, output_dir=str(evidence_dir))
    print(f"Evidencia versionada en {evidence_dir / 'model_comparison.json'}")
    print(f"Evidencia MLflow en {mlflow_evidence}")

    best_model = max(comparison, key=lambda name: comparison[name]["f1_macro"])
    print(f"Mejor f1_macro: {best_model} ({comparison[best_model]['f1_macro']:.4f})")

    print(f"\n=== Actualizando artefactos DVC con {best_model} (mejor f1_macro) ===")
    best_params = copy.deepcopy(params)
    best_params["model"]["type"] = best_model
    train_model(best_params, write_dvc_outputs=True)


if __name__ == "__main__":
    main()
