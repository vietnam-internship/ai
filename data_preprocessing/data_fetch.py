import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine

#.env 파일 불러오기
load_dotenv()

#DB 연결을 관리할 전역 변수
_engine: Optional[Engine] = None


def get_engine(port_num:str = "3306") -> Engine:
    global _engine
    if _engine is None:
        host = os.environ["DB_HOST"]
        port = os.environ.get("DB_PORT", port_num)
        user = os.environ["DB_USER"]
        password = os.environ["DB_PASSWORD"]
        name = os.environ["DB_NAME"]
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}"
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def fetch_exchange_rate_timeseries(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cur_unit: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    engine = engine or get_engine()

    conditions = []
    params: dict = {}

    if start_date:
        conditions.append("erh.recorded_at >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("erh.recorded_at <= :end_date")
        params["end_date"] = end_date
    # 특정 통화의 환율만 조회
    if cur_unit:
        conditions.append("c.code = :cur_unit")
        params["cur_unit"] = cur_unit

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = text(
        f"""
        SELECT erh.recorded_at AS date, c.code AS cur_unit, erh.rate AS rate
        FROM exchange_rate_history erh
        JOIN currency c ON c.id = erh.currency_id
        {where_clause}
        ORDER BY erh.recorded_at ASC
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])

    return df[["date", "cur_unit", "rate"]]
