
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline import build_baseline_dataset
from data_preprocessing.feature import (
    calculate_lagged_rates,
    calculate_moving_average,
    calculate_moving_std,
)
from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries

#feature를 만들 때 쓰는 이동평균/표준편차 기간
WINDOW = 30
#1일전, 2일전, 3일전 환율값을 feature로 씀
DEFAULT_LAGS = (1, 2, 3)


def check_row_num(df: pd.DataFrame, window: int = WINDOW, horizon: int = 1) -> Optional[pd.DataFrame]:
    if len(df) < window + horizon:
        return None
    return df



def build_features(
    df: pd.DataFrame,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    df = calculate_moving_average(df, window=window)
    df = calculate_moving_std(df, window=window)
    df = calculate_lagged_rates(df, lags=lags)
    return df


def feature_columns(window: int = WINDOW, lags: Iterable[int] = DEFAULT_LAGS) -> list[str]:
    return ["rate", f"rate_ma{window}", f"rate_std{window}", *(f"rate_lag{lag}" for lag in lags)]


#X: rate, ma30, std30, lag1~3, y: horizon일 뒤 환율 - 오늘 환율(rate_diff)
def build_diff_dataset(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> Tuple[pd.DataFrame, pd.Series]:
    
    if check_row_num(df, window=window, horizon=horizon) is None:
        raise ValueError("데이터가 충분하지 않습니다.")

    #feature 생성
    df = build_features(df, window=window, lags=lags)
    cols = feature_columns(window=window, lags=lags)

    #target(y) 생성
    diff_groups = []
    for _, group in df.groupby("cur_unit", sort=False):
        group = group.sort_values("date").copy()
        group["rate_diff"] = group["rate"].shift(-horizon) - group["rate"]
        diff_groups.append(group)
    df = pd.concat(diff_groups).sort_index()

    #결측치 제거 및 반환
    df = df.dropna(subset=[*cols, "rate_diff"])

    X = df[cols].reset_index(drop=True)
    y = df["rate_diff"].reset_index(drop=True)
    return X, y


def train_lr_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


#시계열 데이터 validation 방식
def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = WINDOW,
) -> Tuple[pd.Series, pd.Series]:
    
    if len(X) <= train_window:
        raise ValueError("데이터가 충분하지 않습니다.")

    actual_values = []
    predicted_values = []
    for i in range(train_window, len(X)):
        X_train = X.iloc[i - train_window : i]
        y_train = y.iloc[i - train_window : i]

        model = train_lr_model(X_train, y_train)
        predicted_values.append(model.predict(X.iloc[[i]])[0])
        actual_values.append(y.iloc[i])

    return (
        pd.Series(actual_values, name="actual"),
        pd.Series(predicted_values, name="predicted"),
    )


def build_and_walk_forward_validate(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> Tuple[pd.Series, pd.Series]:
    
    actual_parts = []
    predicted_parts = []

    #통화별로 독립적인 feature를 생성 -> validation 실행
    for _, group in df.groupby("cur_unit", sort=False):
        try:
            X, y = build_diff_dataset(group, horizon=horizon, window=window, lags=lags)
        except ValueError:
            continue
        if len(X) <= train_window:
            continue

        actual, predicted = walk_forward_validate(X, y, train_window=train_window)
        actual_parts.append(actual)
        predicted_parts.append(predicted)

    if not actual_parts:
        raise ValueError("데이터가 충분하지 않습니다.")

    return (
        pd.concat(actual_parts, ignore_index=True),
        pd.concat(predicted_parts, ignore_index=True),
    )


def _metrics_from_predictions(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }



def _fit_latest_window_model(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> LinearRegression:
    
    X_parts = []
    y_parts = []
    #통화별로 최근 데이터만 수집
    for _, group in df.groupby("cur_unit", sort=False):
        try:
            X, y = build_diff_dataset(group, horizon=horizon, window=window, lags=lags)
        except ValueError:
            continue
        X_parts.append(X.tail(train_window))
        y_parts.append(y.tail(train_window))


    if not X_parts:
        raise ValueError("데이터가 충분하지 않습니다.")

    return train_lr_model(pd.concat(X_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True))


#전체 파이프라인 실행
def build_and_train(
    df: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> Tuple[LinearRegression, dict]:
    # df가 없다면 DB에서 직접 환율 데이터 가져옴
    if df is None:
        df = fetch_and_fill_exchange_rate_timeseries()

    actual, predicted = build_and_walk_forward_validate(
        df, horizon=horizon, window=window, lags=lags, train_window=train_window
    )
    metrics = _metrics_from_predictions(actual, predicted)

    model = _fit_latest_window_model(df, horizon=horizon, window=window, lags=lags, train_window=train_window)
    return model, metrics


#학습된 모델로 현재 시점 기준, 앞으로 환율이 얼마나 변할지 예측하는 함수
def predict_diff(model: LinearRegression, df: pd.DataFrame, window: int = WINDOW, lags: Iterable[int] = DEFAULT_LAGS) -> pd.Series:
    
    #feature 생성
    df = build_features(df, window=window, lags=lags)
    cols = feature_columns(window=window, lags=lags)

    latest = df.dropna(subset=cols).groupby("cur_unit", sort=False).tail(1)
    X_latest = latest[cols].reset_index(drop=True)

    predicted_diff = model.predict(X_latest)
    return pd.Series(predicted_diff, index=latest["cur_unit"].values, name="predicted_diff")


#baseline(다음날 환율 = 오늘까지의 30일 이동평균)의 MAE. 데이터가 부족하면 None
def baseline_mae(df: pd.DataFrame) -> Optional[float]:
    try:
        X, y = build_baseline_dataset(df)
    except ValueError:
        return None
    if len(y) == 0:
        return None
    return float((y - X["rate_pred"]).abs().mean())


#baseline 예측을 LR과 같은 diff(예측 환율 - 오늘 환율) 형태로 변환
def baseline_predict_diff(df: pd.DataFrame) -> pd.Series:
    with_ma = calculate_moving_average(df, window=WINDOW)
    latest = with_ma.dropna(subset=[f"rate_ma{WINDOW}"]).groupby("cur_unit", sort=False).tail(1)
    predicted_diff = latest[f"rate_ma{WINDOW}"] - latest["rate"]
    return pd.Series(predicted_diff.values, index=latest["cur_unit"].values, name="predicted_diff")


#LR을 학습하되, 데이터가 부족해 학습이 안 되거나 baseline보다 성능(MAE)이 나쁘면 baseline으로 대체한다.
def build_and_train_with_fallback(
    df: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> dict:
    if df is None:
        df = fetch_and_fill_exchange_rate_timeseries()

    baseline_mae_value = baseline_mae(df)

    try:
        model, lr_metrics = build_and_train(
            df=df, horizon=horizon, window=window, lags=lags, train_window=train_window
        )
    except ValueError:
        # walk-forward validation에 필요한 최소 행 수(train_window + window + horizon 근처)를
        # 채우지 못해 학습 자체가 불가능한 경우
        return {
            "source": "baseline",
            "reason": "insufficient_data",
            "model": None,
            "metrics": {"mae": baseline_mae_value},
            "predicted_diff": baseline_predict_diff(df) if baseline_mae_value is not None else None,
            "lr_mae": None,
            "baseline_mae": baseline_mae_value,
        }

    if baseline_mae_value is not None and lr_metrics["mae"] >= baseline_mae_value:
        # LR이 학습은 됐지만 "그냥 30일 평균을 쓰는 것"보다도 못 맞히는 경우
        return {
            "source": "baseline",
            "reason": "worse_than_baseline",
            "model": None,
            "metrics": {"mae": baseline_mae_value},
            "predicted_diff": baseline_predict_diff(df),
            "lr_mae": lr_metrics["mae"],
            "baseline_mae": baseline_mae_value,
        }

    return {
        "source": "lr",
        "reason": None,
        "model": model,
        "metrics": lr_metrics,
        "predicted_diff": predict_diff(model, df, window=window, lags=lags),
        "lr_mae": lr_metrics["mae"],
        "baseline_mae": baseline_mae_value,
    }
