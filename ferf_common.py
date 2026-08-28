
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "binary_label"

NON_PREDICTIVE = {
    "binary_label",
    "multiclass_label",
    "original_label",
    "source_dataset",
    "source_file",
    "source_row",
}

IDENTIFIER_PATTERNS = (
    r"(^|_)flow_id($|_)",
    r"(^|_)src_ip($|_)",
    r"(^|_)dst_ip($|_)",
    r"(^|_)source_ip($|_)",
    r"(^|_)destination_ip($|_)",
    r"(^|_)record_id($|_)",
)

TIMESTAMP_PATTERNS = (
    r"(^|_)timestamp($|_)",
    r"(^|_)time_stamp($|_)",
    r"(^|_)stime($|_)",
    r"(^|_)ltime($|_)",
)

VIEW_TOKENS = {
    "volume": (
        "byte", "packet", "length", "len", "load", "rate",
        "total", "mean", "avg", "min", "max", "std",
    ),
    "temporal": (
        "duration", "dur", "iat", "active", "idle", "time", "jitter",
    ),
    "transport": (
        "protocol", "proto", "service", "state", "port",
        "sport", "dport", "window", "header",
    ),
    "flags_errors": (
        "flag", "fin", "syn", "rst", "psh", "ack", "urg",
        "ece", "cwr", "error", "loss", "retrans",
    ),
    "directional": (
        "fwd", "forward", "bwd", "backward", "source",
        "destination", "src", "dst", "inbound", "outbound",
    ),
}


def normalize_name(value: Any) -> str:
    text = re.sub(
        r"[^a-z0-9_]+",
        "_",
        str(value).strip().lower(),
    )
    return re.sub(r"_+", "_", text).strip("_")


def matches_pattern(column: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_name(column)
    return any(re.search(pattern, normalized) for pattern in patterns)


def select_predictors(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if TARGET not in dataframe.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")

    target = pd.to_numeric(
        dataframe[TARGET],
        errors="coerce",
    )

    valid = target.isin([0, 1])
    dataframe = dataframe.loc[valid].copy()
    target = target.loc[valid].astype("int8")

    excluded = set(NON_PREDICTIVE)

    for column in dataframe.columns:
        if (
            matches_pattern(column, IDENTIFIER_PATTERNS)
            or matches_pattern(column, TIMESTAMP_PATTERNS)
        ):
            excluded.add(column)

    features = dataframe[
        [
            column
            for column in dataframe.columns
            if column not in excluded
        ]
    ].copy()

    removable = [
        column
        for column in features.columns
        if (
            features[column].isna().all()
            or features[column].nunique(dropna=True) <= 1
        )
    ]

    if removable:
        features = features.drop(columns=removable)

    if features.empty:
        raise ValueError("No predictive features remained.")

    return features, target, sorted(excluded | set(removable))


def build_views(
    features: pd.DataFrame,
    minimum_features: int = 2,
) -> dict[str, list[str]]:
    assigned: set[str] = set()
    views: dict[str, list[str]] = {}

    for view_name, tokens in VIEW_TOKENS.items():
        columns = sorted(
            {
                column
                for column in features.columns
                if any(
                    token in normalize_name(column)
                    for token in tokens
                )
            }
        )

        if len(columns) >= minimum_features:
            views[view_name] = columns
            assigned.update(columns)

    residual = [
        column
        for column in features.columns
        if column not in assigned
    ]

    if len(residual) >= minimum_features:
        views["general"] = residual

    if len(views) < 2:
        raise ValueError(
            "FERF requires at least two non-empty evidence views."
        )

    return views


def create_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    numeric = list(
        features.select_dtypes(
            include=[np.number, "bool"]
        ).columns
    )

    categorical = [
        column
        for column in features.columns
        if column not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric,
            )
        )

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=10,
                        sparse_output=True,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical,
            )
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )


def create_xgboost(
    seed: int = 42,
    estimators: int = 250,
):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def calculate_metrics(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    target_array = np.asarray(target)
    prediction = (
        np.asarray(probability) >= threshold
    ).astype("int8")

    return {
        "accuracy": float(
            accuracy_score(target_array, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target_array,
                prediction,
            )
        ),
        "precision": float(
            precision_score(
                target_array,
                prediction,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target_array,
                prediction,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target_array,
                prediction,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                target_array,
                prediction,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target_array,
                probability,
            )
        ),
        "average_precision": float(
            average_precision_score(
                target_array,
                probability,
            )
        ),
    }


class QualityEstimator:
    """
    Label-free, training-fitted information-quality estimator.
    """

    def fit(
        self,
        features: pd.DataFrame,
    ) -> "QualityEstimator":
        self.columns_ = list(features.columns)

        self.numeric_ = list(
            features.select_dtypes(
                include=[np.number, "bool"]
            ).columns
        )

        self.categorical_ = [
            column
            for column in self.columns_
            if column not in self.numeric_
        ]

        self.q01_ = (
            features[self.numeric_].quantile(0.01)
            if self.numeric_
            else pd.Series(dtype=float)
        )

        self.q99_ = (
            features[self.numeric_].quantile(0.99)
            if self.numeric_
            else pd.Series(dtype=float)
        )

        self.valid_categories_ = {
            column: set(
                features[column]
                .dropna()
                .astype(str)
                .value_counts()
                .head(500)
                .index
            )
            for column in self.categorical_
        }

        return self

    def transform(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        features = features[self.columns_]

        observed_fraction = (
            1.0
            - features.isna().mean(axis=1).to_numpy()
        )

        if self.numeric_:
            numeric = features[self.numeric_].apply(
                pd.to_numeric,
                errors="coerce",
            )

            plausible = (
                (
                    (
                        numeric.ge(self.q01_, axis=1)
                        & numeric.le(self.q99_, axis=1)
                    )
                    | numeric.isna()
                )
                .mean(axis=1)
                .to_numpy()
            )
        else:
            plausible = np.ones(len(features))

        if self.categorical_:
            validity_rows = []

            for column in self.categorical_:
                values = features[column]

                validity_rows.append(
                    (
                        values.isna()
                        | values.astype(str).isin(
                            self.valid_categories_[column]
                        )
                    )
                    .astype(float)
                    .to_numpy()
                )

            category_validity = np.vstack(
                validity_rows
            ).mean(axis=0)
        else:
            category_validity = np.ones(
                len(features)
            )

        return np.clip(
            0.50 * observed_fraction
            + 0.35 * plausible
            + 0.15 * category_validity,
            0.0,
            1.0,
        )


def integrity_scores(
    dataset_root: Path,
    source_files: pd.Series,
) -> np.ndarray:
    manifest = (
        dataset_root
        / "Manifests"
        / "source_file_manifest.json"
    )

    if not manifest.exists():
        return np.full(
            len(source_files),
            0.5,
            dtype=float,
        )

    try:
        data = json.loads(
            manifest.read_text(
                encoding="utf-8"
            )
        )

        verified_files = {
            Path(record["source_file"]).name
            for record in data.get(
                "source_files",
                [],
            )
            if record.get("sha256")
        }

        return (
            source_files
            .astype(str)
            .map(
                lambda value: (
                    1.0
                    if Path(value).name
                    in verified_files
                    else 0.5
                )
            )
            .to_numpy(dtype=float)
        )

    except Exception:
        return np.full(
            len(source_files),
            0.5,
            dtype=float,
        )


def temporal_scores(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    timestamp_columns = [
        column
        for column in dataframe.columns
        if matches_pattern(
            column,
            TIMESTAMP_PATTERNS,
        )
    ]

    if not timestamp_columns:
        return np.full(
            len(dataframe),
            0.5,
            dtype=float,
        )

    parsed = pd.to_datetime(
        dataframe[timestamp_columns[0]],
        errors="coerce",
        utc=True,
    )

    return (
        parsed.notna()
        .astype(float)
        .to_numpy()
    )


def save_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
