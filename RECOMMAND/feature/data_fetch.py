import re
from typing import Optional

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from data_preprocessing.data_fetch import get_engine

#branches.business_hours 자유 텍스트("평일 09:00-18:00, 토 09:00-13:00")를 파싱하는 정규식.
#BusinessHoursParser.java(백엔드)와 동일한 포맷/토큰만 인식한다.
_BUSINESS_HOURS_SEGMENT = re.compile(r"(평일|주말|월|화|수|목|금|토|일)\s+(\d{2}:\d{2})-(\d{2}:\d{2})")
_WEEKDAY_TOKENS: dict[str, list[int]] = {
    "평일": [0, 1, 2, 3, 4],
    "주말": [5, 6],
    "월": [0],
    "화": [1],
    "수": [2],
    "목": [3],
    "금": [4],
    "토": [5],
    "일": [6],
}


def _parse_business_hours(business_hours: Optional[str]) -> dict[int, tuple[str, str]]:
    """day_of_week(0=월~6=일) -> (open_time, close_time) HH:MM:SS 매핑. 매칭 안 되는 요일은 휴무로 본다."""
    hours: dict[int, tuple[str, str]] = {}
    if not business_hours:
        return hours

    for segment in business_hours.split(","):
        match = _BUSINESS_HOURS_SEGMENT.match(segment.strip())
        if not match:
            continue
        token, start, end = match.group(1), match.group(2), match.group(3)
        for day in _WEEKDAY_TOKENS.get(token, []):
            hours[day] = (f"{start}:00", f"{end}:00")
    return hours


def fetch_branch_candidates(
    currency_code: str,
    slot_date: str,
    slot_time: str,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """예약 가능 정원이 남아있는 지점 후보를 가져온다.

    주의: 실제 스키마엔 일반 재고(remaining) 컬럼이 없다 (branch_currency_rates엔
    reservation_only_stock만 존재). PRD §18의 w3(재고/availability_score)는 원본 데이터가
    없어 스코어링에서 제외했다 (heuristic.DEFAULT_WEIGHTS 참고, distance/rate/reservation
    3요소로 재분배)."""
    engine = engine or get_engine()

    query = text("""
        SELECT
            b.id AS branch_id,
            b.name AS branch_name,
            b.latitude,
            b.longitude,
            bcr.preferential_rate,
            bcr.reservation_only_stock,
            c.buy_rate,
            c.sell_rate,
            bts.id AS time_slot_id,
            b.time_slot_capacity AS slot_capacity,
            bts.remaining AS slot_remaining
        FROM branches b
        JOIN branch_currency_rates bcr
            ON bcr.branch_id = b.id
            AND bcr.currency_code = :currency_code
        JOIN currencies c
            ON c.code = bcr.currency_code
        JOIN branch_time_slots bts
            ON bts.branch_id = b.id
            AND bts.slot_date = :slot_date
            AND bts.slot_time = :slot_time
            AND bts.remaining > 0
        WHERE b.active = TRUE
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "currency_code": currency_code,
                "slot_date": slot_date,
                "slot_time": slot_time,
            },
        )

    return df


def fetch_branch_operating_hours(
    branch_ids: list[int],
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """heuristic.is_open_now가 기대하는 (branch_id, day_of_week, open_time, close_time, is_closed)
    행 구조를, 실제 스키마의 branches.business_hours 자유 텍스트를 파싱해서 만들어낸다.
    (실제 백엔드엔 요일별 구조화 테이블 자체가 없다.)"""
    engine = engine or get_engine()

    columns = ["branch_id", "day_of_week", "open_time", "close_time", "is_closed"]
    if not branch_ids:
        return pd.DataFrame(columns=columns)

    query = text("""
        SELECT id AS branch_id, business_hours
        FROM branches
        WHERE id IN :branch_ids
    """).bindparams(bindparam("branch_ids", expanding=True))

    with engine.connect() as conn:
        branches_df = pd.read_sql(query, conn, params={"branch_ids": list(branch_ids)})

    rows = []
    for _, row in branches_df.iterrows():
        parsed = _parse_business_hours(row["business_hours"])
        for day in range(7):
            if day in parsed:
                open_time, close_time = parsed[day]
                rows.append(
                    {
                        "branch_id": row["branch_id"],
                        "day_of_week": day,
                        "open_time": open_time,
                        "close_time": close_time,
                        "is_closed": False,
                    }
                )
            else:
                rows.append(
                    {
                        "branch_id": row["branch_id"],
                        "day_of_week": day,
                        "open_time": "00:00:00",
                        "close_time": "00:00:00",
                        "is_closed": True,
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def fetch_branch_recommendation_logs(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Phase 2(로지스틱 리그레션) 학습용 클릭/전환 로그.

    실제 스키마엔 is_selected 컬럼이 없다 — branch_recommendation_items(추천 세션의 지점별
    점수 breakdown)에 branch_recommendation_clicks(클릭 이벤트)가 LEFT JOIN으로 존재하는지로
    선택 여부를 판단한다 (클릭 로그가 있으면 선택된 것)."""
    engine = engine or get_engine()

    query = text("""
        SELECT
            i.distance_score,
            i.rate_score,
            i.reservation_score,
            (c.id IS NOT NULL) AS is_selected
        FROM branch_recommendation_items i
        LEFT JOIN branch_recommendation_clicks c
            ON c.recommendation_item_id = i.id
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df
