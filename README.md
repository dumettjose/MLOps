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
3. `train_model` → modelo, metricas y reportes

Los datos y artefactos pesados no van en Git; DVC los versiona mediante `dvc.lock`.

## Modelos de clasificacion

El proyecto entrena y compara **tres clasificadores**:

| `model.type` | Algoritmo |
|--------------|-----------|
| `random_forest` | RandomForestClassifier |
| `logistic_regression` | LogisticRegression |
| `lightgbm` | LGBMClassifier |

### Entrenar y comparar los modelos

```bash
python -m poetry run python -m src.generate_data --params params.yaml
python -m poetry run python -m src.prepare_data --params params.yaml
python -m poetry run python -m src.train_all --params params.yaml
```

Esto registra experimentos en MLflow y genera evidencias en `evidence/`:

- `evidence/model_comparison.json` — metricas por modelo
- `evidence/mlflow_runs.json` — resumen de runs MLflow

### Cambiar el modelo por defecto del pipeline DVC

Edita `params.yaml`:

```yaml
model:
  type: random_forest   # random_forest | logistic_regression | lightgbm
```

Luego ejecuta:

```bash
python -m poetry run dvc repro
```

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
