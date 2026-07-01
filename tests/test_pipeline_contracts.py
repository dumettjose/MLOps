from src.generate_data import build_mock_sessions
from src.train import TARGET_COLUMN, evaluate, get_feature_columns


def test_mock_dataset_has_expected_target() -> None:
    data = build_mock_sessions(
        n_samples=50,
        random_state=42,
        score_noise_std=2.5,
        label_noise=0.04,
    )

    assert TARGET_COLUMN in data.columns
    assert set(data[TARGET_COLUMN].unique()).issubset({"low", "medium", "high"})
    assert len(data) == 50


def test_preprocessed_features_exclude_target() -> None:
    sample = {
        "historical_avg_session_minutes": [10.0],
        "historical_sessions_last_7d": [3],
        "days_since_last_session": [5],
        "hour_of_day": [12],
        "day_of_week": [2],
        "push_received_last_24h": [1],
        "segment_active": [1],
        TARGET_COLUMN: ["high"],
    }
    import pandas as pd

    data = pd.DataFrame(sample)
    feature_columns = get_feature_columns(data)

    assert TARGET_COLUMN not in feature_columns
    assert len(feature_columns) >= 6


def test_classification_metrics_contract() -> None:
    metrics = evaluate(["low", "medium", "high"], ["low", "high", "high"])

    assert set(metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert all(isinstance(value, float) for value in metrics.values())
