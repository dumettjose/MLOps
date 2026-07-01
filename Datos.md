# Parte de datos (Luis)

Pipeline de generacion y preparacion del dataset para clasificacion de `engagement_level`.

## Target

| Columna | Valores |
|---------|---------|
| `engagement_level` | `low`, `medium`, `high` |

## Etapas DVC

1. **generate_data** → `data/raw/session_data.csv`
2. **prepare_data** → `data/processed/train.csv`, `data/processed/test.csv`

## Ejecutar

```bash
python -m poetry install
python -m poetry run dvc repro
```

O manualmente:

```bash
python -m poetry run python -m src.generate_data --params params.yaml
python -m poetry run python -m src.prepare_data --params params.yaml
```

## Parametros (`params.yaml`)

Solo esta rama controla las secciones `data:` y `split:`.

## Salidas

```text
data/raw/session_data.csv
data/processed/train.csv
data/processed/test.csv
```

Los CSV no van en Git; DVC los versiona via `dvc.lock`.
