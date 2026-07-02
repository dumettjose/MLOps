import argparse

import numpy as np
import pandas as pd

from .config import ensure_parent, load_params


SEGMENTS = ["new", "active", "at_risk", "churn"]
DEVICE_OS = ["android", "ios", "web"]
SITES = ["home", "search", "product", "content"]
ENTRY_POINTS = ["home", "search", "recommendation", "notification"]
ENGAGEMENT_LEVELS = ["low", "medium", "high"]

# Probabilidad de recibir un push de re-enganche por segmento.
# Las campañas reales se dirigen mas a usuarios en riesgo / con churn,
# no de forma uniforme como en la version original.
PUSH_PROB_BY_SEGMENT = {"new": 0.30, "active": 0.40, "at_risk": 0.60, "churn": 0.70}

# Percentiles usados para cortar el score latente en clases.
# No son 33/33/33: reflejan un funnel de engagement realista
# (muchos usuarios "low", pocos "high").
LOW_MEDIUM_CUTOFF = 0.55
MEDIUM_HIGH_CUTOFF = 0.85


def build_mock_sessions(
    n_samples: int,
    random_state: int,
    score_noise_std: float,
    label_noise: float,
) -> pd.DataFrame:
    """Genera un dataset mock reproducible para clasificacion de nivel de engagement.

    En vez de un target continuo (session_minutes), se construye un score latente
    de engagement a partir de las mismas variables de negocio y se discretiza en
    3 clases (low / medium / high). Se añade ruido de etiqueta para simular
    errores de anotacion/negocio, algo propio de problemas de clasificacion real.
    """
    rng = np.random.default_rng(random_state)

    # Variable latente de "propension al engagement": no se expone como feature,
    # pero correlaciona varias columnas observables entre si (como pasa en datos reales).
    latent_engagement = rng.normal(0.0, 1.0, size=n_samples)

    segment = rng.choice(SEGMENTS, size=n_samples, p=[0.25, 0.45, 0.2, 0.1])

    historical_avg_session_minutes = np.clip(
        8.0 + 5.0 * latent_engagement + rng.normal(0.0, 4.0, size=n_samples),
        1.0,
        60.0,
    )

    sessions_lambda = np.clip(3.0 + 1.5 * latent_engagement, 0.1, None)
    historical_sessions_last_7d = rng.poisson(sessions_lambda)

    days_since_last_session = np.clip(
        15.0 - 6.0 * latent_engagement + rng.normal(0.0, 5.0, size=n_samples),
        0,
        30,
    ).astype(int)

    hour_of_day = rng.integers(0, 24, size=n_samples)
    day_of_week = rng.integers(0, 7, size=n_samples)
    device_os = rng.choice(DEVICE_OS, size=n_samples, p=[0.45, 0.35, 0.2])
    site = rng.choice(SITES, size=n_samples)
    entry_point = rng.choice(ENTRY_POINTS, size=n_samples)

    push_prob = pd.Series(segment).map(PUSH_PROB_BY_SEGMENT).to_numpy()
    push_received_last_24h = rng.binomial(1, push_prob)

    data = pd.DataFrame(
        {
            "segment": segment,
            "historical_avg_session_minutes": historical_avg_session_minutes.round(2),
            "historical_sessions_last_7d": historical_sessions_last_7d,
            "days_since_last_session": days_since_last_session,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "device_os": device_os,
            "site": site,
            "entry_point": entry_point,
            "push_received_last_24h": push_received_last_24h,
        }
    )

    segment_effect = data["segment"].map(
        {"new": -2.5, "active": 4.0, "at_risk": -4.0, "churn": -7.5}
    )
    entry_effect = data["entry_point"].map(
        {"home": 1.5, "search": 2.0, "recommendation": 5.5, "notification": 3.0}
    )
    site_effect = data["site"].map({"home": 1.0, "search": 1.5, "product": 2.5, "content": 3.0})
    evening_effect = np.where(data["hour_of_day"].between(18, 22), 3.0, 0.0)
    weekend_effect = np.where(data["day_of_week"].isin([5, 6]), 2.0, 0.0)
    noise = rng.normal(0.0, score_noise_std, size=n_samples)

    engagement_score = (
        6.0
        + 0.58 * data["historical_avg_session_minutes"]
        + 0.85 * data["historical_sessions_last_7d"]
        - 0.18 * data["days_since_last_session"]
        + 2.7 * data["push_received_last_24h"]
        + 2.0 * latent_engagement
        + segment_effect
        + entry_effect
        + site_effect
        + evening_effect
        + weekend_effect
        + noise
    )

    low_cut, high_cut = np.quantile(engagement_score, [LOW_MEDIUM_CUTOFF, MEDIUM_HIGH_CUTOFF])
    engagement_level = np.select(
        [engagement_score < low_cut, engagement_score < high_cut],
        ["low", "medium"],
        default="high",
    )

    # Ruido de etiqueta: una fraccion de filas recibe una clase aleatoria distinta
    # a la "verdadera". Simula errores de anotacion / ambiguedad de negocio y evita
    # que el target sea una funcion deterministica de las features (irreducible error).
    if label_noise > 0:
        flip_mask = rng.random(n_samples) < label_noise
        n_flips = int(flip_mask.sum())
        if n_flips > 0:
            random_levels = rng.choice(ENGAGEMENT_LEVELS, size=n_flips)
            engagement_level = engagement_level.copy()
            engagement_level[flip_mask] = random_levels

    data["engagement_level"] = engagement_level
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_params = params["data"]

    data = build_mock_sessions(
        n_samples=int(data_params["n_samples"]),
        random_state=int(data_params["random_state"]),
        score_noise_std=float(data_params["score_noise_std"]),
        label_noise=float(data_params.get("label_noise", 0.0)),
    )

    output_path = ensure_parent(data_params["raw_path"])
    data.to_csv(output_path, index=False)
    print(f"Dataset mock creado en {output_path} con {len(data)} filas.")
    print(data["engagement_level"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
