import os
from datetime import date
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
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine

def fetch_currency(engine:Optional[Engine]=None) ->list[int]:
    engine = engine or get_engine()
    
    query = text(
        f"""SELECT id, code, country, buy_rate, sell_rate, updated_at
            FROM currencies"""
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=["updated_at"])
    
    return df

def fetch_exchange_rate_history(
    currency_id:int,
    start_date:Optional[str]=None,
    end_date:Optional[str]=None,
    engine:Optional[Engine]=None,
    ) -> pd.DataFrame:
    
    engine = engine or get_engine()

    if end_date is None:
        end_date = str(date.today())

    conditions = ["currency_id = :currency_id", "recorded_at <= :end_date"]
    params = {"currency_id": currency_id, "end_date": end_date}
    if start_date is not None:
        conditions.append("recorded_at >= :start_date")
        params["start_date"] = start_date

    query = text(f"""
        SELECT id, currency_id, rate, recorded_at
        FROM exchange_rate_histories
        WHERE {" AND ".join(conditions)}
        ORDER BY recorded_at ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(
            query, conn,
            params=params,
            parse_dates=["recorded_at"],
        )

    return df

def preprocess_all_currency(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> dict[int, pd.DataFrame]:
    engine = engine or get_engine()

    currencies_df = fetch_currency(engine=engine)
    results = {}

    for _, row in currencies_df.iterrows():
        currency_id = row["id"]
        history_df = fetch_exchange_rate_history(
            currency_id=currency_id,
            start_date=start_date,
            end_date=end_date,
            engine=engine,
        )
        results[currency_id] = history_df

    return results
