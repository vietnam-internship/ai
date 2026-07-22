#학습된 LR 모델을 실제 서비스(live inference)에서 쓰기 위한 인터페이스.
#save_model/load_model로 재학습 없이 모델을 재사용하고, predict_latest로 가장 최근 시점 기준
#통화별 예측(diff, 예측 환율, INCREASING/DECREASI 방향, 근거 feature인 이동평균/표준편차)을 만든다.
#model을 안 넘기면 baseline(30일 이동평균)으로 대체한다.
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from data_preprocessing.feature import calculate_moving_average, calculate_moving_std
from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
from model.lr_model import DEFAULT_LAGS, WINDOW, baseline_predict_diff, predict_diff


def save_model(model: LinearRegression, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> LinearRegression:
    return joblib.load(path)


#통화별 가장 최근 시점의 환율, 예측 diff, 예측 환율, 방향(UP/DOWN), 이동평균/표준편차를 담은 DataFrame을 만든다.
#UP: predicted_rate > current_rate, DOWN: predicted_rate < current_rate, 같으면 FLAT
#model을 넘기면 LR로, 안 넘기면 baseline(30일 이동평균)으로 예측한다.
#df를 넘기지 않으면 DB에서 최신 데이터를 가져온다.
def predict_latest(
    model: Optional[LinearRegression] = None,
    df: Optional[pd.DataFrame] = None,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    if df is None:
        df = fetch_and_fill_exchange_rate_timeseries()

    if model is not None:
        predicted_diff = predict_diff(model, df, window=window, lags=lags)
        source = "lr"
    else:
        predicted_diff = baseline_predict_diff(df)
        source = "baseline"

    latest_rate = (
        df.sort_values("date")
        .groupby("cur_unit", sort=False)["rate"]
        .last()
        .reindex(predicted_diff.index)
    )

    ma_col, std_col = f"rate_ma{window}", f"rate_std{window}"
    with_features = calculate_moving_std(calculate_moving_average(df, window=window), window=window)
    latest_features = (
        with_features.sort_values("date")
        .groupby("cur_unit", sort=False)[[ma_col, std_col]]
        .last()
        .reindex(predicted_diff.index)
    )

    result = pd.DataFrame({"current_rate": latest_rate, "predicted_diff": predicted_diff})
    result["moving_avg"] = latest_features[ma_col]
    result["moving_std"] = latest_features[std_col]
    result["predicted_rate"] = result["current_rate"] + result["predicted_diff"]
    result["direction"] = np.select(
        [result["predicted_rate"] > result["current_rate"], result["predicted_rate"] < result["current_rate"]],
        ["INCREASING","DECREASING"],
        default="NEUTRAL",
    )
    result["source"] = source
    result.index.name = "cur_unit"
    return result
