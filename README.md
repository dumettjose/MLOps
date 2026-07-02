# MLOps — Clasificacion de engagement_level

Proyecto end-to-end de **clasificacion multiclass** para predecir el nivel de engagement de usuarios (`low`, `medium`, `high`). El flujo integra **Poetry**, **DVC** y **MLflow**.

## Equipo

Ver `integrantes.md` para nombres, roles y responsabilidades.

## Requisitos

- Python `>=3.14`
- [Poetry](https://python-poetry.org/)
- Git

Opcional en macOS para LightGBM:

```bash
brew install libomp
```

## Instalacion

Clona el repositorio y entra a la carpeta del proyecto:

```bash
git clone https://github.com/dumettjose/MLOps.git
cd MLOps
```

Crea el entorno e instala dependencias:

```bash
python -m poetry install
```

Poetry crea el entorno virtual en `.venv/` (configurado en `poetry.toml`).

## Comandos minimos esperados

Bloque minimo de ejecucion requerido para el proyecto:

```bash
poetry install
poetry run dvc pull
poetry run dvc repro
poetry run python -m src.train --params params.yaml
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Notas:

- `dvc pull` recupera datos versionados desde el remoto DVC (`./dvc_remote`). Si es la primera ejecucion y aun no hay datos en el remoto, continua con `dvc repro`.
- `dvc repro` entrena **automaticamente los 3 modelos** (`random_forest`, `logistic_regression`, `lightgbm`) y actualiza `evidence/`.
- El comando `src.train` sirve para depurar un solo modelo; el flujo oficial usa `train_all` via DVC.
- Abre MLflow en http://127.0.0.1:5000

Verifica la instalacion:

```bash
python -m poetry run python -c "import src.train; print('ok')"
python -m poetry run pytest
```

## Estructura del proyecto

```text
MLOps/
├── src/
│   ├── generate_data.py      # Generacion de datos
│   ├── prepare_data.py       # Preparacion y split
│   ├── train.py              # Entrenamiento de un modelo
│   ├── train_all.py          # Entrenamiento comparativo
│   └── export_mlflow_evidence.py
├── params.yaml               # Parametros del pipeline
├── dvc.yaml                  # Pipeline DVC
├── dvc.lock                  # Versionamiento DVC
├── pyproject.toml
├── poetry.lock
├── poetry.toml
├── evidence/                 # Evidencias versionadas en Git
├── Datos.md
├── Modelos.md
└── integrantes.md
```

## Ejecucion del pipeline completo (DVC)

Inicializa DVC si aun no existe:

```bash
python -m poetry run dvc init --subdir
```

Ejecuta las 3 etapas:

```bash
python -m poetry run dvc repro
```

Etapas:

1. `generate_data` → `data/raw/session_data.csv`
2. `prepare_data` → `data/processed/train.csv`, `data/processed/test.csv`
3. `train_model` → entrena **los 3 modelos**, registra MLflow y genera evidencias en `evidence/`

Al finalizar `dvc repro` se actualizan automaticamente en disco:

- `evidence/model_comparison.json` — metricas de los 3 modelos
- `evidence/mlflow_runs.json` — resumen de experimentos MLflow
- `models/classification_model.joblib` — mejor modelo segun `f1_macro`

Estos archivos en `evidence/` se versionan en Git para la entrega del proyecto.

## Modelos de clasificacion

El pipeline entrena **automaticamente tres clasificadores** en cada `dvc repro`:

| Modelo | Algoritmo |
|--------|-----------|
| `random_forest` | RandomForestClassifier |
| `logistic_regression` | LogisticRegression |
| `lightgbm` | LGBMClassifier |

No es necesario cambiar `params.yaml` entre modelos: `train_all.py` los ejecuta en secuencia.

Evidencias generadas en `evidence/`:

- `evidence/model_comparison.json` — metricas por modelo
- `evidence/mlflow_runs.json` — resumen de runs MLflow

### Entrenamiento manual (opcional)

Si quieres repetir solo la etapa de modelos sin DVC:

```bash
python -m poetry run python -m src.train_all --params params.yaml
```

### Depurar un solo modelo

```bash
python -m poetry run python -m src.train --params params.yaml
```

En ese caso define temporalmente `model.type` en `params.yaml` (`random_forest`, `logistic_regression` o `lightgbm`).

## MLflow

Levanta la interfaz local:

```bash
python -m poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000

Para exportar evidencias versionables en Git:

```bash
python -m poetry run python -m src.export_mlflow_evidence --params params.yaml
```

## Metricas evaluadas

- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`

## Que versionar en Git

Versionar:

```text
README.md
integrantes.md
pyproject.toml
poetry.lock
poetry.toml
params.yaml
dvc.yaml
dvc.lock
.dvc/
src/
tests/
evidence/
```

No versionar:

```text
data/raw/
data/processed/
models/
reports/
mlruns/
mlflow.db
.venv/
```

## Documentacion adicional

- `Datos.md` — modulo de generacion y preparacion de datos
- `Modelos.md` — modulo de entrenamiento y evaluacion

## Pruebas

```bash
python -m poetry run pytest
```
