from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from customer_churn.dataset import MODEL_FEATURES

TARGET = "churned_in_personal_window"


@dataclass
class TrainingResult:
    selected_name: str
    selected_model: Pipeline
    calibrator: LogisticRegression
    test_frame: pd.DataFrame
    test_probabilities: np.ndarray
    comparison: pd.DataFrame
    metrics: dict[str, object]
    calibration: pd.DataFrame
    feature_importance: pd.DataFrame


def _time_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(dataset["score_date"]).dt.normalize().unique()))
    if len(dates) < 10:
        raise ValueError("At least ten distinct scoring dates are required for a time split.")
    train_end = dates[max(1, int(len(dates) * 0.60)) - 1]
    calibration_end = dates[max(2, int(len(dates) * 0.80)) - 1]
    train = dataset.loc[pd.to_datetime(dataset["score_date"]) <= train_end].copy()
    calibration = dataset.loc[
        (pd.to_datetime(dataset["score_date"]) > train_end)
        & (pd.to_datetime(dataset["score_date"]) <= calibration_end)
    ].copy()
    test = dataset.loc[pd.to_datetime(dataset["score_date"]) > calibration_end].copy()
    if min(train[TARGET].nunique(), calibration[TARGET].nunique(), test[TARGET].nunique()) < 2:
        raise ValueError("Every time split must contain both target classes.")
    return train, calibration, test


def _probability_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def _top_fraction_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    y = np.asarray(y_true)
    count = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-probabilities)[:count]
    prevalence = float(y.mean())
    precision = float(y[selected].mean())
    recall = float(y[selected].sum() / max(y.sum(), 1))
    return {
        "precision": precision,
        "recall": recall,
        "lift": precision / max(prevalence, 1e-12),
    }


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def train_and_evaluate(dataset: pd.DataFrame) -> TrainingResult:
    """Train interpretable and nonlinear models with chronological holdouts."""

    train, calibration_frame, test = _time_split(dataset)
    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET]
    x_calibration = calibration_frame[MODEL_FEATURES]
    y_calibration = calibration_frame[TARGET]
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET]

    models: dict[str, Pipeline] = {
        "Logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=180,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    comparison_rows = [
        {
            "model": "Fixed 90-day recency rule",
            **_probability_metrics(
                y_calibration,
                np.clip(calibration_frame["current_gap_days"].to_numpy() / 90.0, 0, 1),
            ),
        },
        {
            "model": "Personalized window rule",
            **_probability_metrics(
                y_calibration,
                np.clip(calibration_frame["window_progress"].to_numpy(), 0, 1),
            ),
        },
    ]
    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_calibration)[:, 1]
        fitted[name] = model
        validation_probabilities[name] = probabilities
        comparison_rows.append(
            {"model": name, **_probability_metrics(y_calibration, probabilities)}
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False)
    selected_name = max(
        fitted,
        key=lambda name: average_precision_score(
            y_calibration, validation_probabilities[name]
        ),
    )
    selected_model = fitted[selected_name]
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(_logit(validation_probabilities[selected_name]), y_calibration)
    raw_test_probabilities = selected_model.predict_proba(x_test)[:, 1]
    test_probabilities = calibrator.predict_proba(_logit(raw_test_probabilities))[:, 1]

    fraction_metrics = {
        "top_10_percent": _top_fraction_metrics(y_test, test_probabilities, 0.10),
        "top_20_percent": _top_fraction_metrics(y_test, test_probabilities, 0.20),
    }
    test_metrics = {
        **_probability_metrics(y_test, test_probabilities),
        "brier_score": float(brier_score_loss(y_test, test_probabilities)),
        **fraction_metrics,
    }
    probability_true, probability_predicted = calibration_curve(
        y_test,
        test_probabilities,
        n_bins=10,
        strategy="quantile",
    )
    calibration_table = pd.DataFrame(
        {
            "mean_predicted_probability": probability_predicted,
            "observed_churn_rate": probability_true,
        }
    )
    importance = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_pr_auc_decrease": importance.importances_mean,
            "std_pr_auc_decrease": importance.importances_std,
        }
    ).sort_values("mean_pr_auc_decrease", ascending=False)
    metrics: dict[str, object] = {
        "selected_model": selected_name,
        "split": {
            "train_rows": int(len(train)),
            "calibration_rows": int(len(calibration_frame)),
            "test_rows": int(len(test)),
            "train_end": str(pd.to_datetime(train["score_date"]).max().date()),
            "calibration_end": str(
                pd.to_datetime(calibration_frame["score_date"]).max().date()
            ),
            "test_end": str(pd.to_datetime(test["score_date"]).max().date()),
        },
        "test": test_metrics,
    }
    return TrainingResult(
        selected_name=selected_name,
        selected_model=selected_model,
        calibrator=calibrator,
        test_frame=test,
        test_probabilities=test_probabilities,
        comparison=comparison,
        metrics=metrics,
        calibration=calibration_table,
        feature_importance=feature_importance,
    )
