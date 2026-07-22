import numpy as np
from datetime import date, timedelta
from typing import Optional
import pandas as pd
from data_preprocessing.data_fetch import fetch_currency, preprocess_all_currency

#lookback_days를 명시하지 않은 호출부를 위한 기본값. 무제한 fetch(DB 전체 이력)를 막기 위한
#안전장치. 모델이 실제로 필요한 최소 기간(window+train_window+horizon)은 이 값과 무관하게
#호출하는 쪽(model.lr_model 등)이 자기 요구사항에 맞춰 lookback_days를 직접 넘기면 된다.
DEFAULT_LOOKBACK_DAYS = 120

#NaN이 아니거나, min_rate보다 작은 환율은 거른다. 
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

        modified_z = 0.6745 * (rate - rmedian) / mad.replace(0, pd.NA)

        keep = modified_z.isna() | (modified_z.abs() <= threshold)
        keep_masks.append(keep)

    keep_mask = pd.concat(keep_masks).sort_index()
    return df.loc[keep_mask].reset_index(drop=True)


def validate_exchange_rates(
    df: pd.DataFrame,
    min_rate: float = 0,
    window: int = 20,
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
    lookback_days: Optional[int] = DEFAULT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    #start_date를 직접 안 넘겼으면 lookback_days만큼만 최근 데이터를 가져온다.
    #DB 전체 이력이 필요한 특수 케이스는 lookback_days=None으로 명시적으로 풀어준다.
    if start_date is None and lookback_days is not None:
        start_date = str(date.today() - timedelta(days=lookback_days))

    history_by_currency_id = preprocess_all_currency(
        start_date=start_date,
        end_date=end_date,
    )
    currencies_df = fetch_currency()
    code_by_id = dict(zip(currencies_df["id"], currencies_df["code"]))

    frames = []
    for currency_id, history_df in history_by_currency_id.items():
        if history_df.empty:
            continue
        frame = history_df.rename(columns={"recorded_at": "date"})
        frame["cur_unit"] = code_by_id.get(currency_id, currency_id)
        frames.append(frame[["date", "cur_unit", "rate"]])

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["date", "cur_unit", "rate"])
    )

    if cur_unit:
        df = df[df["cur_unit"] == cur_unit].reset_index(drop=True)

    df = validate_exchange_rates(df)
    return fill_missing_dates(df)
