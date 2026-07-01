# Parte de modelos (José)

Clasificacion multiclass de `engagement_level` (`low`, `medium`, `high`).

## Modelos soportados

| `model.type` | Algoritmo |
|--------------|-----------|
| `random_forest` | RandomForestClassifier |
| `logistic_regression` | LogisticRegression + StandardScaler |
| `lightgbm` | LGBMClassifier |

## Instalacion

```bash
python -m poetry install
```

En macOS, si LightGBM falla:

```bash
brew install libomp
```

## Pipeline completo (DVC)

```bash
python -m poetry run dvc repro
```

Entrena el modelo definido en `params.yaml` (por defecto `random_forest`).

## Entrenar los 3 modelos y comparar

```bash
python -m poetry run python -m src.generate_data --params params.yaml
python -m poetry run python -m src.prepare_data --params params.yaml
python -m poetry run python -m src.train_all --params params.yaml
```

Genera:

- 3 runs en MLflow (uno por modelo)
- `reports/model_comparison.json` con metricas comparadas
- Artefactos por modelo en `reports/{modelo}_*`
- Artefactos DVC del modelo por defecto en `reports/metrics.json`, etc.

## MLflow UI

```bash
python -m poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir http://127.0.0.1:5000

## Cambiar modelo por defecto (DVC)

En `params.yaml`:

```yaml
model:
  type: random_forest   # random_forest | logistic_regression | lightgbm
  random_state: 60
  n_estimators: 200
  max_depth: 10
  max_iter: 1000        # logistic_regression
  learning_rate: 0.1    # lightgbm
  class_weight: balanced
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
