from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from RECOMMAND.feature.heuristic import DEFAULT_WEIGHTS

FEATURE_COLUMNS = ["distance_score", "rate_score", "availability_score", "reservation_score"]
TARGET_COLUMN = "is_selected"

MIN_SAMPLES = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_dataset(logs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = logs[FEATURE_COLUMNS].reset_index(drop=True)
    y = logs[TARGET_COLUMN].reset_index(drop=True)
    return X, y


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

#계수를 가중치로 변환
def coefficients_to_weights(model: LogisticRegression) -> Optional[dict]:
    coefs = model.coef_[0]
    if np.any(coefs < 0):
        return None

    total = coefs.sum()
    if total <= 0:
        return None

    return dict(zip(("distance", "rate", "availability", "reservation"), (coefs / total).tolist()))


def weighted_sum_scores(X: pd.DataFrame, weights: dict) -> pd.Series:
    return (
        weights["distance"] * X["distance_score"]
        + weights["rate"] * X["rate_score"]
        + weights["availability"] * X["availability_score"]
        + weights["reservation"] * X["reservation_score"]
    )


def _classification_metrics(y_true: pd.Series, y_score: pd.Series) -> dict:
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = None

    clipped = np.clip(y_score, 1e-6, 1 - 1e-6)
    return {"auc": auc, "logLoss": float(log_loss(y_true, clipped))}


#Logistic Regression 모델이 baseline(DEFAULT_WEIGHTS 가중합)보다 AUC는 높고 log loss는 낮아야 "더 낫다"고 인정한다.
#AUC를 판단할 수 없는 경우(단일 클래스 held-out) log loss만으로 비교한다.
def is_better_than_baseline(lr_metrics: dict, baseline_metrics: dict) -> bool:
    if lr_metrics["auc"] is not None and baseline_metrics["auc"] is not None:
        if lr_metrics["auc"] <= baseline_metrics["auc"]:
            return False
    return lr_metrics["logLoss"] < baseline_metrics["logLoss"]


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = train_logistic_regression(X_train, y_train)
    lr_scores = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "lr_metrics": _classification_metrics(y_test, lr_scores),
        "baseline_metrics": _classification_metrics(y_test, weighted_sum_scores(X_test, DEFAULT_WEIGHTS)),
    }


def _fallback_result(reason: str, train_rows: int, metrics: Optional[dict] = None) -> dict:
    return {
        "source": "default",
        "reason": reason,
        "weights": DEFAULT_WEIGHTS,
        "model": None,
        "metrics": metrics or {},
        "trainRows": train_rows,
    }


#지점 추천 가중치를 학습하되, 신뢰할 수 없으면(로그 부족/음수 계수/baseline보다 못함)
#DEFAULT_WEIGHTS로 폴백한 결과를 반환한다. logs가 None이면 DB에서 직접 가져온다.
def build_and_train_with_fallback(
    logs: Optional[pd.DataFrame] = None,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    #log 데이터가 없으면 fetch
    if logs is None:
        from RECOMMAND.feature.data_fetch import fetch_branch_recommendation_logs

        logs = fetch_branch_recommendation_logs()
    #log 데이터가 10개 미만이면 fallback
    if logs.empty or len(logs) < min_samples:
        return _fallback_result("insufficient_data", train_rows=len(logs))

    X, y = build_dataset(logs)
    if y.nunique() < 2:
        return _fallback_result("insufficient_data", train_rows=len(logs))

    try:
        evaluation = train_and_evaluate(X, y)
    except ValueError:
        # stratified split이 클래스별 최소 표본 수를 만족하지 못하는 경우
        return _fallback_result("insufficient_data", train_rows=len(logs))

    lr_metrics = evaluation["lr_metrics"]
    baseline_metrics = evaluation["baseline_metrics"]
    metrics = {"lr": lr_metrics, "baseline": baseline_metrics}

    weights = coefficients_to_weights(evaluation["model"])
    if weights is None:
        return _fallback_result("negative_coefficient", train_rows=len(logs), metrics=metrics)

    if not is_better_than_baseline(lr_metrics, baseline_metrics):
        return _fallback_result("worse_than_baseline", train_rows=len(logs), metrics=metrics)

    return {
        "source": "logistic_regression",
        "reason": None,
        "weights": weights,
        "model": evaluation["model"],
        "metrics": metrics,
        "trainRows": len(logs),
    }
