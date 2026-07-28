"""LR(+baseline fallback) 환율 타이밍 예측을, 백엔드에 push할 페이로드로 변환한다.

model.inference.predict_latest가 model=None이면 자동으로 baseline(30일 이동평균)을
쓰므로, LR 아티팩트가 아직 없는 통화도 그대로 baseline으로 서빙된다 (PRD §9.4 fallback).
"""
from datetime import datetime, timedelta, timezone

import numpy as np

from api.config import DEFAULT_RECOMMENDATION_TTL_HOURS
from api.errors import InsufficientDataError, ModelUnavailableError
from api.model_registry import load_lr_model
from api.schemas import (
    AiRecommendationCreateRequest,
    AiSignalCreateRequest,
    RecommendationEnum,
    SignalTypeEnum,
)
from model.lr_model import WINDOW

_DIRECTION_TO_RECOMMENDATION = {
    "INCREASING": RecommendationEnum.INCREASING,
    "DECREASING": RecommendationEnum.DECREASING,
    "NEUTRAL": RecommendationEnum.NEUTRAL,
}


def _confidence_score(current_rate: float, moving_std: float) -> float:
    """변동성이 클수록 확신도가 낮다는 단순 휴리스틱.

    LR 실패/신뢰도 판정 기준이 아직 협의되지 않아(infra 이슈 체크리스트), 우선
    "최근 표준편차 / 현재 환율" 비율을 [0, 1]로 clamp해서 쓴다. 협의 결과가 나오면 교체.
    """
    if current_rate is None or current_rate <= 0 or moving_std is None or np.isnan(moving_std):
        return 0.5
    volatility_ratio = moving_std / current_rate
    return float(max(0.0, min(1.0, 1 - volatility_ratio)))


def build_timing_recommendation(
    currency_code: str,
) -> tuple[AiRecommendationCreateRequest, list[AiSignalCreateRequest]]:
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
    from model.inference import predict_latest

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=currency_code)
    if df.empty:
        raise InsufficientDataError(f"{currency_code} 환율 데이터가 없습니다.")

    model, model_version = load_lr_model(currency_code)

    try:
        result = predict_latest(model=model, df=df)
    except ValueError as exc:
        # baseline_predict_diff 등이 "데이터가 충분하지 않습니다"를 명시적으로 던지는 경우
        raise InsufficientDataError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 예기치 못한 추론 실패
        raise ModelUnavailableError(f"예측 실패: {exc}") from exc

    if currency_code not in result.index:
        # LR 경로에서 최소 window+horizon 데이터가 없으면 예외 없이 빈 결과로 돌아온다.
        raise InsufficientDataError(f"{currency_code}에 대한 예측 결과가 없습니다.")

    row = result.loc[currency_code]
    source = row["source"]
    version = model_version if source == "lr" else "baseline-ma30"

    confidence = _confidence_score(row["current_rate"], row["moving_std"])
    recommendation = _DIRECTION_TO_RECOMMENDATION[row["direction"]]

    pct_change = (row["predicted_rate"] - row["current_rate"]) / row["current_rate"] * 100
    rationale = (
        f"Based on {WINDOW}-day moving average/std ({'LR' if source == 'lr' else 'baseline'} model), "
        f"rate expected to change {pct_change:+.2f}% from current {row['current_rate']:.2f}"
    )

    recommendation_payload = AiRecommendationCreateRequest(
        recommendation=recommendation,
        rationale=rationale,
        confidenceScore=confidence,
        modelVersion=version,
        expiresAt=datetime.now(timezone.utc) + timedelta(hours=DEFAULT_RECOMMENDATION_TTL_HOURS),
    )

    signals_payload = [
        AiSignalCreateRequest(signalType=SignalTypeEnum.MA, windowDays=WINDOW, value=float(row["moving_avg"])),
        AiSignalCreateRequest(signalType=SignalTypeEnum.STD, windowDays=WINDOW, value=float(row["moving_std"])),
    ]

    return recommendation_payload, signals_payload
