# Equipo MLOps

Proyecto de clasificacion de `engagement_level` con Poetry, DVC y MLflow.

| Integrante | Rol | Responsabilidades |
|------------|-----|-------------------|
| Luis Jimenez | Datos | Generacion del dataset mock, limpieza, split train/test, etapas DVC `generate_data` y `prepare_data`, `params.yaml` (secciones `data` y `split`) |
| Jose Diaz Dumett | Modelos | Entrenamiento y evaluacion de clasificadores, integracion con MLflow, etapa DVC `train_model`, comparacion de modelos, pruebas de la parte de modelos |
| Sergio Esteban Leon Garcia | Integracion | Gestion de ramas en GitHub, integracion del trabajo del equipo, documentacion general del repositorio y apoyo en la entrega del proyecto |

## Ramas de trabajo

| Rama | Responsable principal |
|------|------------------------|
| `feature/data-preparation` | Luis Jimenez |
| `feature/model-training` | Jose Diaz Dumett |
| `main` | Integracion del equipo |

## Documentacion por modulo

- `Datos.md` — pipeline de datos
- `Modelos.md` — pipeline de modelos
- `evidence/` — evidencias de experimentos MLflow y comparacion de modelos
