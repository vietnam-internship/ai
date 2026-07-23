"""DB/백엔드 서버 없이 목데이터로 /internal/ai/train, /internal/ai/predict 를 돌려보는 스크립트.

실제로는:
- data_preprocessing.preprocess.fetch_and_fill_exchange_rate_timeseries 가 MySQL을 직접 읽고
- api.routers.predict 가 실제 백엔드(BACKEND_BASE_URL)로 push한다.

여기서는 두 지점을 가짜 함수로 바꿔치기(monkeypatch)해서, 그 사이의 로직
(LR/baseline 학습, 추천 계산, 라우터/스키마 유효성 검증)만 검증한다.

사용법 (ai/ 디렉토리에서 실행):
    python3 scripts/mock_run.py
"""
import os
import sys
import tempfile
from pathlib import Path

# ai/ 를 import 루트로 추가 (어느 위치에서 실행해도 동작하도록)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# api.config가 import되기 전에 설정해야 실제 model/artifacts를 건드리지 않는다.
os.environ.setdefault("MODEL_ARTIFACT_DIR", tempfile.mkdtemp(prefix="travelx-ai-mock-"))

import numpy as np
import pandas as pd

import data_preprocessing.preprocess as preprocess_module


def _mock_exchange_rate_df(days: int = 200, cur_unit: str = "USD") -> pd.DataFrame:
    """DB 대신 쓸 가짜 환율 이력 (완만한 상승 추세 + 노이즈가 섞인 랜덤워크)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    rate = 1380 + np.cumsum(rng.normal(loc=0.05, scale=3.0, size=days))
    return pd.DataFrame({"date": dates, "cur_unit": cur_unit, "rate": rate})


_MOCK_DF = _mock_exchange_rate_df()


def _fake_fetch(*args, cur_unit=None, **kwargs):
    if cur_unit:
        return _MOCK_DF[_MOCK_DF["cur_unit"] == cur_unit].reset_index(drop=True)
    return _MOCK_DF.copy()


# train.py / services/timing.py 는 함수 안에서 그때그때
# `from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries` 를 실행하므로,
# 모듈 속성 자체를 바꿔치기해두면 두 곳 모두에 적용된다.
preprocess_module.fetch_and_fill_exchange_rate_timeseries = _fake_fetch

import data_preprocessing.data_fetch as data_fetch_module
import RECOMMAND.feature.data_fetch as branch_data_fetch_module

_MOCK_CURRENCY_DF = pd.DataFrame([{"id": 1, "code": "USD", "country": "미국"}])
_MOCK_BRANCH_CANDIDATES_DF = pd.DataFrame(
    [
        {
            "branch_id": 1,
            "branch_name": "강남점",
            "latitude": 37.5,
            "longitude": 127.0,
            "preferential_rate": 0.01,
            "currency_remaining": 500,
            "reservation_only_stock": 50,
            "buy_rate": 1300.0,
            "sell_rate": 1320.0,
            "time_slot_id": 10,
            "slot_capacity": 5,
            "slot_remaining": 3,
        },
    ]
)
_MOCK_OPERATING_HOURS_DF = pd.DataFrame(
    [{"branch_id": 1, "day_of_week": d, "open_time": "09:00:00", "close_time": "18:00:00", "is_closed": False} for d in range(7)]
)


def _mock_branch_feedback_df(rows: int = 150) -> pd.DataFrame:
    """Phase 2(로지스틱 리그레션) 학습 임계값(train.MIN_BRANCH_FEEDBACK_ROWS)을 넘기는 가짜 클릭/전환
    로그. availability_score가 높을수록 is_selected 확률도 높아지게 만들어서, 학습된 가중치가
    availability_score 쪽에 실리는지 눈으로 확인할 수 있게 한다."""
    rng = np.random.default_rng(0)
    availability_score = rng.random(rows)
    return pd.DataFrame(
        {
            "distance_score": rng.random(rows),
            "rate_score": rng.random(rows),
            "availability_score": availability_score,
            "reservation_score": rng.random(rows),
            "is_selected": rng.random(rows) < availability_score,
        }
    )


data_fetch_module.fetch_currency = lambda *a, **k: _MOCK_CURRENCY_DF.copy()
branch_data_fetch_module.fetch_branch_candidates = lambda *a, **k: _MOCK_BRANCH_CANDIDATES_DF.copy()
branch_data_fetch_module.fetch_branch_operating_hours = lambda *a, **k: _MOCK_OPERATING_HOURS_DF.copy()
branch_data_fetch_module.fetch_branch_recommendation_logs = lambda *a, **k: _mock_branch_feedback_df()

from fastapi.testclient import TestClient

from api.main import app
import api.routers.predict as predict_router_module

# predict.py는 파일 맨 위에서 push_recommendation/push_signals를 직접 import해두므로,
# backend_client 쪽이 아니라 predict 모듈에 이미 바인딩된 이름을 바꿔야 실제로 적용된다.
predict_router_module.push_recommendation = lambda code, payload: {"id": 1, "mock": True}
predict_router_module.push_signals = lambda code, signals: [
    {"id": i + 1, "mock": True} for i in range(len(signals))
]

client = TestClient(app)


def show(title: str, response) -> None:
    print(f"\n=== {title} ===")
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    show("health (학습 전)", client.get("/internal/ai/health"))

    show(
        "train: BASE_LINE",
        client.post("/internal/ai/train", json={"strategyType": "BASE_LINE", "currencyCode": "USD"}),
    )

    show(
        "train: LINEAR_REGRESSION",
        client.post("/internal/ai/train", json={"strategyType": "LINEAR_REGRESSION", "currencyCode": "USD"}),
    )

    show("health (학습 후)", client.get("/internal/ai/health"))

    show("predict", client.post("/internal/ai/currencies/USD/predict"))

    branch_recommend_request = {
        "currencyCode": "USD",
        "userLat": 37.5,
        "userLng": 127.0,
        "slotDate": str(pd.Timestamp.today().date()),
        "slotTime": "10:00:00",
    }

    show(
        "branches/recommend (Phase 2 학습 전 - weightsSource: default)",
        client.post("/internal/ai/branches/recommend", json=branch_recommend_request),
    )

    show(
        "train: LOGISTIC_REGRESSION (지점 추천 Phase 2, 가짜 로그 150건으로 학습)",
        client.post("/internal/ai/train", json={"strategyType": "LOGISTIC_REGRESSION"}),
    )

    show(
        "branches/recommend (Phase 2 학습 후 - weightsSource: logistic_regression)",
        client.post("/internal/ai/branches/recommend", json=branch_recommend_request),
    )
