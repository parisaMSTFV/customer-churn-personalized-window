from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from customer_churn.dataset import MODEL_FEATURES

TARGET = "churned_in_personal_window"


class RuleClassifier(ClassifierMixin, BaseEstimator):
    """Deployable sklearn-compatible recency baseline."""

    def __init__(self, rule: str = "personalized") -> None:
        self.rule = rule

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RuleClassifier:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.rule == "fixed_90_day":
            probability = np.clip(np.asarray(x["current_gap_days"], dtype=float) / 90.0, 0, 1)
        elif self.rule == "personalized":
            probability = np.clip(np.asarray(x["window_progress"], dtype=float), 0, 1)
        else:
            raise ValueError(f"Unknown rule: {self.rule}")
        return np.column_stack([1 - probability, probability])


@dataclass
class TrainingResult:
    selected_name: str
    selected_model: object
    calibrator: LogisticRegression
    test_frame: pd.DataFrame
    test_probabilities: np.ndarray
    comparison: pd.DataFrame
    metrics: dict[str, object]
    calibration: pd.DataFrame
    feature_importance: pd.DataFrame
    feature_reference: pd.DataFrame


def _time_split(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(dataset["score_date"]).dt.normalize().unique()))
    if len(dates) < 16:
        raise ValueError("At least sixteen distinct scoring dates are required for a time split.")
    train_end = dates[max(1, int(len(dates) * 0.50)) - 1]
    validation_end = dates[max(2, int(len(dates) * 0.70)) - 1]
    calibration_end = dates[max(3, int(len(dates) * 0.85)) - 1]
    score_dates = pd.to_datetime(dataset["score_date"])
    train = dataset.loc[score_dates <= train_end].copy()
    validation = dataset.loc[(score_dates > train_end) & (score_dates <= validation_end)].copy()
    calibration = dataset.loc[
        (score_dates > validation_end) & (score_dates <= calibration_end)
    ].copy()
    test = dataset.loc[score_dates > calibration_end].copy()
    if (
        min(
            train[TARGET].nunique(),
            validation[TARGET].nunique(),
            calibration[TARGET].nunique(),
            test[TARGET].nunique(),
        )
        < 2
    ):
        raise ValueError("Every time split must contain both target classes.")
    return train, validation, calibration, test


def _expected_calibration_error(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y = np.asarray(y_true)
    assignments = pd.qcut(probabilities, bins, labels=False, duplicates="drop")
    error = 0.0
    for index in np.unique(assignments):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(float(y[mask].mean() - probabilities[mask].mean()))
    return error


def _probability_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "expected_calibration_error": _expected_calibration_error(y_true, scores),
    }


def _top_fraction_metrics(
    y_true: pd.Series, probabilities: np.ndarray, fraction: float
) -> dict[str, float]:
    y = np.asarray(y_true)
    count = max(1, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-probabilities)[:count]
    prevalence = float(y.mean())
    precision = float(y[selected].mean())
    recall = float(y[selected].sum() / max(y.sum(), 1))
    return {"precision": precision, "recall": recall, "lift": precision / prevalence}


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _calibration_coefficients(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    model = LogisticRegression(random_state=42)
    model.fit(_logit(probabilities), y_true)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    iterations: int = 200,
) -> dict[str, object]:
    work = frame[["customer_id", TARGET]].copy()
    work["probability"] = probabilities
    groups = {key: value for key, value in work.groupby("customer_id", sort=False)}
    customers = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(42)
    values: dict[str, list[float]] = {"pr_auc": [], "brier_score": []}
    for _ in range(iterations):
        sampled = rng.choice(customers, size=len(customers), replace=True)
        replicate = pd.concat([groups[customer] for customer in sampled], ignore_index=True)
        if replicate[TARGET].nunique() < 2:
            continue
        values["pr_auc"].append(
            float(average_precision_score(replicate[TARGET], replicate["probability"]))
        )
        values["brier_score"].append(
            float(brier_score_loss(replicate[TARGET], replicate["probability"]))
        )
    output: dict[str, object] = {"iterations": iterations, "cluster": "customer"}
    for metric, samples in values.items():
        output[metric] = {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
    return output


def _candidate_models() -> dict[str, object]:
    return {
        "Fixed 90-day recency rule": RuleClassifier("fixed_90_day"),
        "Personalized window rule": RuleClassifier("personalized"),
        "Logistic regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42),
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


def _feature_reference(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in MODEL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        median = float(values.median())
        mad = float((values - median).abs().median())
        rows.append(
            {
                "feature": feature,
                "development_median": median,
                "development_robust_scale": max(1.4826 * mad, 1e-9),
                "development_missing_rate": float(values.isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_drift(
    frame: pd.DataFrame, reference: pd.DataFrame, threshold: float = 2.0
) -> pd.DataFrame:
    """Compare current feature medians with the development reference."""

    rows = []
    for record in reference.to_dict("records"):
        values = pd.to_numeric(frame[record["feature"]], errors="coerce")
        current_median = float(values.median())
        shift = (current_median - record["development_median"]) / record["development_robust_scale"]
        rows.append(
            {
                **record,
                "current_median": current_median,
                "current_missing_rate": float(values.isna().mean()),
                "median_shift_robust_units": float(shift),
                "warning": bool(abs(shift) > threshold),
            }
        )
    return pd.DataFrame(rows)


def score_with_bundle(bundle: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    model = bundle["model"]
    raw = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    calibrator = bundle["calibrator"]
    return calibrator.predict_proba(_logit(raw))[:, 1]


def train_and_evaluate(dataset: pd.DataFrame) -> TrainingResult:
    """Select, refit, calibrate, and test candidates on four chronological periods."""

    train, validation, calibration_frame, test = _time_split(dataset)
    candidates = _candidate_models()
    comparison_rows = []
    for name, candidate in candidates.items():
        fitted = clone(candidate).fit(train[MODEL_FEATURES], train[TARGET])
        probabilities = fitted.predict_proba(validation[MODEL_FEATURES])[:, 1]
        comparison_rows.append(
            {"model": name, **_probability_metrics(validation[TARGET], probabilities)}
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False)
    selected_name = str(comparison.iloc[0]["model"])

    development = pd.concat([train, validation], ignore_index=True)
    selected_model = clone(candidates[selected_name]).fit(
        development[MODEL_FEATURES], development[TARGET]
    )
    calibration_raw = selected_model.predict_proba(calibration_frame[MODEL_FEATURES])[:, 1]
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(_logit(calibration_raw), calibration_frame[TARGET])
    raw_test = selected_model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test_probabilities = calibrator.predict_proba(_logit(raw_test))[:, 1]

    test_metrics: dict[str, object] = {
        **_probability_metrics(test[TARGET], test_probabilities),
        **_calibration_coefficients(test[TARGET], test_probabilities),
        "top_10_percent": _top_fraction_metrics(test[TARGET], test_probabilities, 0.10),
        "top_20_percent": _top_fraction_metrics(test[TARGET], test_probabilities, 0.20),
        "cluster_bootstrap_95": _cluster_bootstrap(test, test_probabilities),
    }
    probability_true, probability_predicted = calibration_curve(
        test[TARGET], test_probabilities, n_bins=10, strategy="quantile"
    )
    calibration_table = pd.DataFrame(
        {
            "mean_predicted_probability": probability_predicted,
            "observed_churn_rate": probability_true,
        }
    )
    importance = permutation_importance(
        selected_model,
        test[MODEL_FEATURES],
        test[TARGET],
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
        "selection_metric": "validation_pr_auc",
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "calibration_rows": int(len(calibration_frame)),
            "test_rows": int(len(test)),
            "train_end": str(pd.to_datetime(train["score_date"]).max().date()),
            "validation_end": str(pd.to_datetime(validation["score_date"]).max().date()),
            "calibration_end": str(pd.to_datetime(calibration_frame["score_date"]).max().date()),
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
        feature_reference=_feature_reference(development),
    )
