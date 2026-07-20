#lr_model의 walk-forward validation 결과(실제 vs 예측 diff)를 MAE/RMSE로 집계한다.
import numpy as np
import pandas as pd

from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
from model.lr_model import build_and_walk_forward_validate


def load_actual_and_predicted() -> tuple[pd.Series, pd.Series]:
    df = fetch_and_fill_exchange_rate_timeseries()
    return build_and_walk_forward_validate(df)


#|실제 diff - 예측 diff| 의 평균
def mae() -> float:
    actual, predicted = load_actual_and_predicted()
    return float((actual - predicted).abs().mean())


#(실제 diff - 예측 diff)^2 의 평균의 제곱근
def rmse() -> float:
    actual, predicted = load_actual_and_predicted()
    return float(np.sqrt(((actual - predicted) ** 2).mean()))
