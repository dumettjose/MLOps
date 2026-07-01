# Parte de modelos (José)

Esta rama contiene el entrenamiento y evaluacion del clasificador de `engagement_level`.

## Contrato con la parte de datos

| Elemento | Valor |
|----------|-------|
| Target | `engagement_level` (`low`, `medium`, `high`) |
| Train | `data/processed/train.csv` |
| Test | `data/processed/test.csv` |
| Features | Todas las columnas excepto el target (ya vienen one-hot encoded desde `prepare_data`) |

## Ejecutar solo la parte de modelos

Primero asegurate de tener los CSV procesados (tu companero los genera con DVC o manualmente):

```bash
python -m poetry install
python -m poetry run python -m src.generate_data --params params.yaml
python -m poetry run python -m src.prepare_data --params params.yaml
python -m poetry run python -m src.train --params params.yaml
```

O el pipeline completo:

```bash
python -m poetry run dvc repro
```

## Salidas

```text
models/classification_model.joblib
reports/metrics.json
reports/predictions.csv
reports/classification_report.txt
reports/confusion_matrix.png
mlflow.db
```

## Metricas

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`

## Cambiar modelo

En `params.yaml`:

```yaml
model:
  type: random_forest   # o logistic_regression
```
