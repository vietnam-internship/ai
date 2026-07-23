from typing import Optional

import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from data_preprocessing.data_fetch import get_engine


def fetch_branch_candidates(
    currency_id: int,
    slot_date: str,
    slot_time: str,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    #재고와 예약 가능 정원이 남아있는 지점 후보를 가져온다.
    engine = engine or get_engine()

    query = text("""
        SELECT
            b.id AS branch_id,
            b.name AS branch_name,
            b.latitude,
            b.longitude,
            bci.preferential_rate,
            bci.remaining AS currency_remaining,
            bci.reservation_only_stock,
            c.buy_rate,
            c.sell_rate,
            bts.id AS time_slot_id,
            bts.capacity AS slot_capacity,
            bts.remaining AS slot_remaining
        FROM branch b
        JOIN branch_currency_inventory bci
            ON bci.branch_id = b.id
            AND bci.currency_id = :currency_id
            AND bci.remaining > 0
        JOIN currency c
            ON c.id = bci.currency_id
        JOIN branch_time_slot bts
            ON bts.branch_id = b.id
            AND bts.slot_date = :slot_date
            AND bts.start_time <= :slot_time
            AND bts.end_time > :slot_time
            AND bts.remaining > 0
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "currency_id": currency_id,
                "slot_date": slot_date,
                "slot_time": slot_time,
            },
        )

    return df


def fetch_branch_operating_hours(
    branch_ids: list[int],
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    engine = engine or get_engine()

    #is_open 계산을 위한 column
    columns = ["branch_id", "day_of_week", "open_time", "close_time", "is_closed"]
    if not branch_ids:
        return pd.DataFrame(columns=columns)

    query = text("""
        SELECT branch_id, day_of_week, open_time, close_time, is_closed
        FROM branch_operating_hours
        WHERE branch_id IN :branch_ids
    """).bindparams(bindparam("branch_ids", expanding=True))

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"branch_ids": list(branch_ids)})

    return df


def fetch_branch_recommendation_logs(engine: Optional[Engine] = None) -> pd.DataFrame:
    """Phase 2(로지스틱 리그레션) 학습용 클릭/전환 로그.

    branches/recommend가 반환한 각 후보(score_candidates의 distance/rate/availability/reservation
    score)와, 사용자가 실제로 그 지점을 선택(예약/전환)했는지(is_selected)를 백엔드가 기록해둔 테이블."""
    engine = engine or get_engine()

    query = text("""
        SELECT distance_score, rate_score, availability_score, reservation_score, is_selected
        FROM branch_recommendation_feedback
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df
