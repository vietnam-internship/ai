#오프라인 백테스팅: DB에 쌓인 실제 환율 이력에 대해 LR 모델을 walk-forward validation으로 검증하고,
#통화별 MAE/RMSE와 baseline(30일 이동평균) 대비 성능을 리포트한다.
#스크립트로 바로 실행하면(`python -m model.backtest`) 콘솔에 리포트를 출력한다.
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
from model.lr_model import (
    DEFAULT_LAGS,
    WINDOW,
    baseline_mae,
    build_diff_dataset,
    walk_forward_validate,
)


#한 통화(cur_unit 하나로만 이루어진 df)에 대해 walk-forward validation을 수행하고 baseline과 비교한다.
def backtest_currency(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> dict:
    X, y = build_diff_dataset(df, horizon=horizon, window=window, lags=lags)
    actual, predicted = walk_forward_validate(X, y, train_window=train_window)
    error = actual - predicted

    #방향(부호) 일치 여부 = "타이밍 추천이 맞았는가"의 신호 단위 정확도.
    total_signals = len(actual)
    correct_signals = int((np.sign(actual) == np.sign(predicted)).sum())

    return {
        "n_predictions": total_signals,
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "baseline_mae": baseline_mae(df),
        "totalSignals": total_signals,
        "correctSignals": correct_signals,
        "accuracyRate": correct_signals / total_signals if total_signals else None,
    }


#df(여러 cur_unit 포함 가능)를 통화별로 백테스트해 리포트 DataFrame을 만든다.
#df를 넘기지 않으면 DB에서 실제 이력을 가져온다(오프라인 백테스트의 기본 진입점).
def run_backtest(
    df: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    window: int = WINDOW,
    lags: Iterable[int] = DEFAULT_LAGS,
    train_window: int = WINDOW,
) -> pd.DataFrame:
    if df is None:
        df = fetch_and_fill_exchange_rate_timeseries()

    rows = {}
    for cur_unit, group in df.groupby("cur_unit", sort=False):
        try:
            rows[cur_unit] = backtest_currency(
                group, horizon=horizon, window=window, lags=lags, train_window=train_window
            )
        except ValueError:
            # 해당 통화는 walk-forward validation에 필요한 데이터가 부족함 -> 리포트에서 제외
            continue

    if not rows:
        raise ValueError("No currency available for backtesting (not enough data).")

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "cur_unit"
    result["beats_baseline"] = result["mae"] < result["baseline_mae"]
    return result


if __name__ == "__main__":
    report = run_backtest()
    print(report.round(4).to_string())
