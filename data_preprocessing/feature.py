from typing import Iterable, Optional
import pandas as pd
from preprocess import fetch_and_fill_exchange_rate_timeseries

#한 개의 윈도우 크기에 대해서 이동평균 계산
def calculate_moving_average(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    
    if df.empty:
        return df

    ma_col = f"{column}_ma{window}"
    ma_groups = []
    for _, group in df.groupby(["source", "cur_unit"], sort=False):
        group = group.sort_values("date").copy()
        group[ma_col] = group[column].rolling(window, min_periods=min_periods).mean()
        ma_groups.append(group)

    result = pd.concat(ma_groups)
    return result.sort_index()

#여러 이동평균 계신 
def calculate_moving_averages(
    df: pd.DataFrame,
    windows: Iterable[int] = (5, 20, 60),
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:

    for window in windows:
        df = calculate_moving_average(df, window=window, column=column, min_periods=min_periods)
    return df


def calculate_moving_std(
    df: pd.DataFrame,
    window: int = 20,
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    
    if df.empty:
        return df

    std_col = f"{column}_std{window}"
    std_groups = []
    for _, group in df.groupby(["source", "cur_unit"], sort=False):
        group = group.sort_values("date").copy()
        group[std_col] = group[column].rolling(window, min_periods=min_periods).std()
        std_groups.append(group)

    result = pd.concat(std_groups)
    return result.sort_index()


def calculate_moving_stds(
    df: pd.DataFrame,
    windows: Iterable[int] = (5, 20, 60),
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """여러 윈도우의 이동표준편차를 한 번에 계산해 컬럼들로 추가한다."""
    for window in windows:
        df = calculate_moving_std(df, window=window, column=column, min_periods=min_periods)
    return df


def fetch_and_calculate_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    cur_unit: Optional[str] = None,
    windows: Iterable[int] = (5, 20, 60),
    column: str = "rate",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """data_fetch -> preprocess를 거친 데이터에 이동평균/이동표준편차를 추가한다."""
    df = fetch_and_fill_exchange_rate_timeseries(
        start_date=start_date,
        end_date=end_date,
        source=source,
        cur_unit=cur_unit,
    )
    df = calculate_moving_averages(df, windows=windows, column=column, min_periods=min_periods)
    df = calculate_moving_stds(df, windows=windows, column=column, min_periods=min_periods)
    return df
