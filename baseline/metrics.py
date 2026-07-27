from typing import Optional

import pandas as pd

from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
from baseline import build_baseline_dataset


def load_actual_and_predicted(df: Optional[pd.DataFrame] = None):
    if df is None:
        df = fetch_and_fill_exchange_rate_timeseries()
    baseX, basey = build_baseline_dataset(df)
    actual = basey
    predicted = baseX['rate_pred']
    return actual, predicted


#(|실제 관측값 - 모델의 예측값| / 실제 관측값 의 합) / 데이터 개수 * 100
def mape(df: Optional[pd.DataFrame] = None):
    actual, predicted = load_actual_and_predicted(df)
    ape = (actual - predicted).abs() / actual
    return ape.mean() * 100


#|실제 관측값 - 모델의 예측값| 의 평균
def mae(df: Optional[pd.DataFrame] = None):
    actual, predicted = load_actual_and_predicted(df)
    return (actual - predicted).abs().mean()
