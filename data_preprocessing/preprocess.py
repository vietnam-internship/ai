import numpy as np
from typing import Optional
import pandas as pd
from data_fetch import fetch_exchange_rate_timeseries

#NaN이 아니거나, min_rate보다 큰 환율만 남김
def remove_invalid_rates(df: pd.DataFrame, min_rate: float = 0) -> pd.DataFrame:
    
    if df.empty:
        return df

    valid = df["rate"].notna() & (df["rate"] > min_rate)
    return df.loc[valid].reset_index(drop=True)


def remove_rate_outliers(
    df: pd.DataFrame,
    window: int = 30,
    threshold: float = 3.0,
    min_periods: int = 5,
) -> pd.DataFrame:
    
    if df.empty:
        return df

    keep_masks = []
    for _, group in df.groupby("cur_unit", sort=False):
        group = group.sort_values("date")
        rate = group["rate"]

        rmedian = rate.rolling(window, min_periods=min_periods, center=True).median()
        mad = (rate - rmedian).abs().rolling(window, min_periods=min_periods, center=True).median()

        # MAD -> 표준편차 근사 스케일 상수(정규분포 가정)
        modified_z = 0.6745 * (rate - rmedian) / mad.replace(0, pd.NA)

        # NaN <= threshold는 False로 평가되어 버리므로, isna()를 먼저 확인해야 함
        keep = modified_z.isna() | (modified_z.abs() <= threshold)
        keep_masks.append(keep)

    keep_mask = pd.concat(keep_masks).sort_index()
    return df.loc[keep_mask].reset_index(drop=True)


def validate_exchange_rates(
    df: pd.DataFrame,
    min_rate: float = 0,
    window: int = 30,
    threshold: float = 3.0,
    min_periods: int = 5,
) -> pd.DataFrame:
    df = remove_invalid_rates(df, min_rate=min_rate)
    df = remove_rate_outliers(df, window=window, threshold=threshold, min_periods=min_periods)
    return df


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    filled_groups = []
    for cur_unit, group in df.groupby("cur_unit", sort=False):
        group = group.set_index("date").sort_index()
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
        group = group.reindex(full_range)

        #ffill() -> forward fill
        group["rate"] = group["rate"].ffill()
        group["cur_unit"] = cur_unit
        group.index.name = "date"
        filled_groups.append(group.reset_index())

    result = pd.concat(filled_groups, ignore_index=True)
    return result.sort_values(["cur_unit", "date"]).reset_index(drop=True)


def fetch_and_fill_exchange_rate_timeseries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cur_unit: Optional[str] = None,
) -> pd.DataFrame:

    df = fetch_exchange_rate_timeseries(
        start_date=start_date,
        end_date=end_date,
        cur_unit=cur_unit,
    )
    df = validate_exchange_rates(df)
    return fill_missing_dates(df)
