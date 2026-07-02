import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ensure_parent, load_params

TARGET_COLUMN = "engagement_level"
CATEGORICAL_COLUMNS = ["segment", "device_os", "site", "entry_point"]


def prepare_dataset(
    raw_path: str,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia, codifica y separa el dataset crudo en train/test.

    - One-hot encoding de variables categoricas, ajustado solo con train
      para evitar fuga de informacion (data leakage) desde test.
    - Split estratificado por el target, ya que las clases estan
      desbalanceadas (ver distribucion low/medium/high en generate_data).
    """
    raw = pd.read_csv(raw_path)

    if raw.isna().any().any():
        na_counts = raw.isna().sum()
        raise ValueError(
            f"El dataset crudo tiene valores nulos, revisa antes de continuar:\n{na_counts[na_counts > 0]}"
        )

    X = raw.drop(columns=[TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    X_train_encoded = pd.get_dummies(X_train, columns=CATEGORICAL_COLUMNS).reset_index(drop=True)
    X_test_encoded = pd.get_dummies(X_test, columns=CATEGORICAL_COLUMNS).reset_index(drop=True)
    
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    train_df = X_train_encoded.copy()
    train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_df = X_test_encoded.copy()
    test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    data_params = params["data"]
    split_params = params["split"]

    train_df, test_df = prepare_dataset(
        raw_path=data_params["raw_path"],
        test_size=float(split_params["test_size"]),
        random_state=int(split_params["random_state"]),
        stratify=bool(split_params.get("stratify", False)),
    )

    train_path = ensure_parent(data_params["processed_train_path"])
    test_path = ensure_parent(data_params["processed_test_path"])
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train guardado en {train_path} con {len(train_df)} filas.")
    print(f"Test guardado en {test_path} con {len(test_df)} filas.")
    print("Distribucion de clases (train):")
    print(train_df[TARGET_COLUMN].value_counts(normalize=True).round(3))
    print("Distribucion de clases (test):")
    print(test_df[TARGET_COLUMN].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()