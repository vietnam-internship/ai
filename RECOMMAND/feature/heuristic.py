from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
DISTANCE_DECAY_TAU_KM = 5.0
RESERVATION_STOCK_FLOOR = 0.05
DEFAULT_RADIUS_KM = 5.0
DEFAULT_TOP_N = 10

# 초기 수동 가중치 (PRD §18: w1=거리, w2=환율, w3=재고, w4=예약)
DEFAULT_WEIGHTS = {
    "distance": 0.35,
    "rate": 0.35,
    "availability": 0.2,
    "reservation": 0.1,
}


# 사용자 위치 - 지점 간 직선 거리(km) 계산
def calculate_distance_km(
    df: pd.DataFrame,
    user_lat: float,
    user_lng: float,
    lat_col: str = "latitude",
    lng_col: str = "longitude",
) -> pd.Series:
    lat1, lng1 = np.radians(user_lat), np.radians(user_lng)
    lat2, lng2 = np.radians(df[lat_col]), np.radians(df[lng_col])

    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


#후보 집합 내에서 0~1로 min-max 정규화. 후보가 1개거나 값이 모두 같으면 0.5로 채운다.
def normalize_min_max(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    span = series.max() - series.min()
    if span == 0 or pd.isna(span):
        return pd.Series(0.5, index=series.index)

    normalized = (series - series.min()) / span
    return normalized if higher_is_better else 1 - normalized


#가까운 지점일수록 1에 가깝도록 거리를 반전 exponential decay로 매끄럽게 감쇠.
def distance_score(
    df: pd.DataFrame,
    user_lat: float,
    user_lng: float,
    tau_km: float = DISTANCE_DECAY_TAU_KM,
    lat_col: str = "latitude",
    lng_col: str = "longitude",
) -> pd.Series:
    distance_km = calculate_distance_km(df, user_lat, user_lng, lat_col=lat_col, lng_col=lng_col)
    return np.exp(-distance_km / tau_km)


#최종 환율 final_rate = benchmark * (1 - preferential_rate)을 후보군 내 min-max 정규화. 매수는 낮을수록, 매도는 높을수록 유리.
def rate_score(df: pd.DataFrame, is_buying: bool = True) -> pd.Series:
    benchmark = df["buy_rate"] if is_buying else df["sell_rate"]
    final_rate = benchmark * (1 - df["preferential_rate"].fillna(0))
    return normalize_min_max(final_rate, higher_is_better=not is_buying)


#TIME 컬럼이 pymysql에서 datetime.time/timedelta 어느 쪽으로 오든 자정 기준 초 단위로 통일해서 비교 가능하게 만든다.
def _seconds_since_midnight(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series.astype(str)).dt.total_seconds()


#branch_operating_hours 원본(day_of_week/open_time/close_time/is_closed)에서 현재 영업 여부를 파생.
#day_of_week는 0(월)~6(일). 해당 지점의 오늘자 행이 없으면 영업 안 함으로 간주.
#표시용(BranchSummary.isOpenNow) 정보이며 total_score 가중합에는 포함되지 않는다.
def is_open_now(
    df: pd.DataFrame,
    operating_hours_df: pd.DataFrame,
    now: Optional[datetime] = None,
    branch_col: str = "branch_id",
) -> pd.Series:
    now = now or datetime.now()
    current_day = now.weekday()
    current_seconds = now.hour * 3600 + now.minute * 60 + now.second

    todays_hours = operating_hours_df[operating_hours_df["day_of_week"] == current_day].set_index(branch_col)

    is_open = (
        (~todays_hours["is_closed"].astype(bool))
        & (_seconds_since_midnight(todays_hours["open_time"]) <= current_seconds)
        & (current_seconds < _seconds_since_midnight(todays_hours["close_time"]))
    )

    return df[branch_col].map(is_open.astype(float)).fillna(0.0)


#지점의 통화 재고(currency_remaining)를 후보군 내 min-max 정규화. 재고가 많을수록 1에 가깝다.
#PRD §18 total_score의 w3(재고, ScoreBreakdown.availabilityScore)에 해당.
def availability_score(df: pd.DataFrame, stock_col: str = "currency_remaining") -> pd.Series:
    return normalize_min_max(df[stock_col].fillna(0), higher_is_better=True)


#예약 전용 재고가 남아있으면 1, 소진되면 RESERVATION_STOCK_FLOOR에 가깝게. 하드 0을 피해 가중합에서 완전히 죽지 않게 한다.
def reservation_score(
    df: pd.DataFrame,
    stock_col: str = "reservation_only_stock",
    floor: float = RESERVATION_STOCK_FLOOR,
) -> pd.Series:
    stock = df[stock_col].fillna(0)
    return pd.Series(np.where(stock > 0, 1.0, floor), index=df.index)


def score_candidates(
    df: pd.DataFrame,
    operating_hours_df: pd.DataFrame,
    user_lat: float,
    user_lng: float,
    is_buying: bool = True,
    now: Optional[datetime] = None,
    weights: Optional[dict] = None,
    radius_km: float = DEFAULT_RADIUS_KM,
    top_n: Optional[int] = DEFAULT_TOP_N,
) -> pd.DataFrame:
    if df.empty:
        return df

    weights = weights or DEFAULT_WEIGHTS
    df = df.copy()

    # radius_km 밖 후보는 스코어링 전에 제외 (PRD §18 검색 반경).
    # 이후 min-max 정규화(rate_score, availability_score)가 이 반경 내 후보군만 대상으로 하도록 필터를 먼저 적용한다.
    distance_km = calculate_distance_km(df, user_lat, user_lng)
    df = df[distance_km <= radius_km].copy()
    if df.empty:
        return df
    distance_km = distance_km.loc[df.index]

    df["distance_score"] = np.exp(-distance_km / DISTANCE_DECAY_TAU_KM)
    df["rate_score"] = rate_score(df, is_buying=is_buying)
    df["availability_score"] = availability_score(df)
    df["reservation_score"] = reservation_score(df)
    df["is_open_now"] = is_open_now(df, operating_hours_df, now=now)

    df["score"] = (
        weights["distance"] * df["distance_score"]
        + weights["rate"] * df["rate_score"]
        + weights["availability"] * df["availability_score"]
        + weights["reservation"] * df["reservation_score"]
    )
    # total_score 내림차순, 동점 시 distanceScore 우선 (PRD §18: 타이브레이커)
    df = df.sort_values(
        ["score", "distance_score"], ascending=[False, False]
    ).reset_index(drop=True)

    return df.head(top_n) if top_n is not None else df
