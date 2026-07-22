from typing import Iterable, Optional
import pandas as pd
from data_preprocessing.preprocess import (
    DEFAULT_LOOKBACK_DAYS,
    fetch_and_fill_exchange_rate_timeseries,
)

#한 개의 윈도우 크기에 대해서 이동평균 계산
def calculate_moving_average(
    df: pd.DataFrame,
    window: int = 30,
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    
    if df.empty:
        return df

    ma_col = f"{column}_ma{window}"
    ma_groups = []
    for _, group in df.groupby("cur_unit", sort=False):
        group = group.sort_values("date").copy()
        group[ma_col] = group[column].rolling(window, min_periods=min_periods).mean()
        ma_groups.append(group)

    result = pd.concat(ma_groups)
    return result.sort_index()

#여러 이동평균 계산
def calculate_moving_averages(
    df: pd.DataFrame,
    windows: Iterable[int] = (7, 30, 60),
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:

    for window in windows:
        df = calculate_moving_average(df, window=window, column=column, min_periods=min_periods)
    return df


def calculate_moving_std(
    df: pd.DataFrame,
    window: int = 30,
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    
    if df.empty:
        return df

    std_col = f"{column}_std{window}"
    std_groups = []
    for _, group in df.groupby("cur_unit", sort=False):
        group = group.sort_values("date").copy()
        group[std_col] = group[column].rolling(window, min_periods=min_periods).std()
        std_groups.append(group)

    result = pd.concat(std_groups)
    return result.sort_index()


def calculate_moving_stds(
    df: pd.DataFrame,
    windows: Iterable[int] = (7, 30, 60),
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    for window in windows:
        df = calculate_moving_std(df, window=window, column=column, min_periods=min_periods)
    return df


#lag일 전 환율 값 (lagged feature)
def calculate_lagged_rate(
    df: pd.DataFrame,
    lag: int = 1,
    column: str = "rate",
) -> pd.DataFrame:

    if df.empty:
        return df

    lag_col = f"{column}_lag{lag}"
    lag_groups = []
    for _, group in df.groupby("cur_unit", sort=False):
        group = group.sort_values("date").copy()
        group[lag_col] = group[column].shift(lag)
        lag_groups.append(group)

    result = pd.concat(lag_groups)
    return result.sort_index()


def calculate_lagged_rates(
    df: pd.DataFrame,
    lags: Iterable[int] = (1, 2, 3),
    column: str = "rate",
) -> pd.DataFrame:
    for lag in lags:
        df = calculate_lagged_rate(df, lag=lag, column=column)
    return df


def fetch_and_calculate_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    currency: Optional[str] = None,
    windows: Iterable[int] = (7, 30, 60),
    lags: Iterable[int] = (1, 2, 3),
    column: str = "rate",
    min_periods: Optional[int] = None,
    lookback_days: Optional[int] = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """data_fetch -> preprocess를 거친 데이터에 이동평균/이동표준편차/lag를 추가한다."""
    df = fetch_and_fill_exchange_rate_timeseries(
        start_date=start_date,
        end_date=end_date,
        cur_unit=currency,
        lookback_days=lookback_days,
    )
    df = calculate_moving_averages(df, windows=windows, column=column, min_periods=min_periods)
    df = calculate_moving_stds(df, windows=windows, column=column, min_periods=min_periods)
    df = calculate_lagged_rates(df, lags=lags, column=column)
    return df
