# MLOps — Clasificacion de engagement_level

Proyecto end-to-end de **clasificacion multiclass** para predecir el nivel de engagement de usuarios (`low`, `medium`, `high`). El flujo integra **Poetry**, **DVC** y **MLflow**.

## Equipo

Ver `integrantes.md` para nombres, roles y responsabilidades.

## Requisitos

| Herramienta | Version / notas |
|-------------|-----------------|
| Python | `>=3.14` ([python.org](https://www.python.org/downloads/)) |
| [Poetry](https://python-poetry.org/docs/#installation) | Gestor de dependencias |
| Git | Control de versiones |
| Homebrew (solo macOS) | Necesario para LightGBM (`libomp`) |

## Guia de ejecucion desde cero

Sigue estos pasos en orden desde una PC nueva (macOS, Linux o Windows con WSL).

### 1. Clonar el repositorio

```bash
git clone https://github.com/Sergiogarcialeo/MLOps.git
cd MLOps
```

### 2. Instalar dependencias con Poetry

```bash
poetry install
```

Poetry crea el entorno virtual en `.venv/` (configurado en `poetry.toml`).

### 3. macOS — instalar LightGBM (obligatorio en Mac)

LightGBM requiere OpenMP. Sin esto, los otros dos modelos entrenan pero `lightgbm` falla.

```bash
# Si no tienes Homebrew: https://brew.sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Activar brew en Apple Silicon (solo la primera vez)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

brew install libomp
```

Verifica:

```bash
poetry run python -c "import lightgbm; print('LightGBM OK')"
```

En **Linux**, LightGBM suele funcionar sin pasos extra. En **Windows**, instala [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) o usa WSL.

### 4. Parche MLflow para Python 3.14 (obligatorio)

MLflow 3.14.0 no arranca la UI en Python 3.14 sin este parche. Ejecutalo **una vez** despues de cada `poetry install`:

```bash
poetry run python scripts/patch_mlflow_py314.py
```

Es seguro ejecutarlo varias veces: el script es idempotente.

### 5. Ejecutar el pipeline completo

```bash
poetry run dvc pull
poetry run dvc repro
```

- `dvc pull` descarga datos del remoto DVC si existe (`./dvc_remote`). Si falla o es la primera vez, continua con `dvc repro`.
- `dvc repro` ejecuta las 3 etapas: generacion de datos, preparacion y entrenamiento de **los 3 modelos**.

Para forzar solo el reentrenamiento de modelos:

```bash
poetry run dvc repro -f train_model
```

### 6. Verificar resultados

```bash
poetry run pytest
cat evidence/model_comparison.json
```

Debes ver metricas de `random_forest`, `logistic_regression` y `lightgbm` sin errores en `"errors"`.

### 7. Abrir MLflow UI

```bash
poetry run python scripts/patch_mlflow_py314.py
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000 en el navegador. Para detener el servidor: `Ctrl+C`.

**Importante:** deja un espacio entre `--backend-store-uri` y `sqlite:///mlflow.db`.

## Comandos minimos esperados

Bloque minimo de ejecucion requerido para el proyecto:

```bash
poetry install
poetry run python scripts/patch_mlflow_py314.py
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
poetry run python -c "import src.train; print('ok')"
poetry run pytest
```

## Solucion de problemas

| Problema | Solucion |
|----------|----------|
| `Library not loaded: libomp.dylib` | `brew install libomp` (macOS) |
| `ImportError: Traversable` al abrir MLflow | `poetry run python scripts/patch_mlflow_py314.py` |
| `IndentationError` en skill_installer | Ejecuta el parche de nuevo (restaura MLflow automaticamente) |
| `--backend-store-urisqlite://...` | Falta espacio: `--backend-store-uri sqlite:///mlflow.db` |
| LightGBM test skipped en pytest | Normal en Mac sin libomp; instala libomp y vuelve a correr |
| `dvc pull` sin remoto configurado | Ignora y usa `dvc repro` directamente |

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
├── scripts/
│   └── patch_mlflow_py314.py # Parche MLflow para Python 3.14
├── evidence/                 # Evidencias versionadas en Git
├── Datos.md
├── Modelos.md
└── integrantes.md
```

## Ejecucion del pipeline completo (DVC)

Inicializa DVC si aun no existe:

```bash
poetry run dvc init --subdir
```

Ejecuta las 3 etapas:

```bash
poetry run dvc repro
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
poetry run python -m src.train_all --params params.yaml
```

### Depurar un solo modelo

```bash
poetry run python -m src.train --params params.yaml
```

En ese caso define temporalmente `model.type` en `params.yaml` (`random_forest`, `logistic_regression` o `lightgbm`).

## MLflow

Levanta la interfaz local (nota el espacio entre `--backend-store-uri` y `sqlite:///`):

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre http://127.0.0.1:5000

Si ves `ImportError: cannot import name 'Traversable' from 'importlib.abc'`, ejecuta el parche de Python 3.14:

```bash
poetry run python scripts/patch_mlflow_py314.py
```

Si bajaste MLflow a una version anterior y la base ya fue creada con 3.14, migra el esquema o regenera `mlflow.db` con `dvc repro`.

Para exportar evidencias versionables en Git:

```bash
poetry run python -m src.export_mlflow_evidence --params params.yaml
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
poetry run pytest
```
