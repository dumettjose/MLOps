# Parte de modelos (José)

Clasificacion multiclass de `engagement_level` (`low`, `medium`, `high`).

## Modelos soportados

| Modelo | Algoritmo |
|--------|-----------|
| `random_forest` | RandomForestClassifier |
| `logistic_regression` | LogisticRegression + StandardScaler |
| `lightgbm` | LGBMClassifier |

## Entrenamiento automatico

La etapa DVC `train_model` ejecuta `src.train_all`, que:

1. Entrena los **3 modelos** sin cambiar `params.yaml`.
2. Registra cada run en MLflow.
3. Escribe evidencias en `evidence/model_comparison.json` y `evidence/mlflow_runs.json`.
4. Guarda como artefacto principal el modelo con mejor `f1_macro`.

```bash
python -m poetry run dvc repro
```

## Parametros compartidos (`params.yaml`)

Los hiperparametros en `model:` aplican a los tres clasificadores:

```yaml
model:
  random_state: 60
  n_estimators: 200
  max_depth: 10
  max_iter: 1000        # logistic_regression
  learning_rate: 0.1    # lightgbm
  class_weight: balanced
```

## MLflow UI

```bash
python -m poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Metricas

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`

## Pruebas

```bash
python -m poetry run pytest
```
