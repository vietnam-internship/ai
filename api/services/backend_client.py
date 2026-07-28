"""openapi.yaml의 기존 AI Internal push 엔드포인트를 호출하는 얇은 HTTP 클라이언트."""
import httpx

from api.config import BACKEND_BASE_URL, BACKEND_REQUEST_TIMEOUT_SECONDS, INTERNAL_AI_TOKEN
from api.schemas import (
    AiRecommendationCreateRequest,
    AiSignalCreateRequest,
    BacktestCreateRequest,
    BranchRecommendationPushRequest,
)


def _headers() -> dict:
    headers = {}
    if INTERNAL_AI_TOKEN:
        headers["Authorization"] = f"Bearer {INTERNAL_AI_TOKEN}"
    return headers


def push_recommendation(currency_code: str, payload: AiRecommendationCreateRequest) -> dict:
    url = f"{BACKEND_BASE_URL}/internal/ai/currencies/{currency_code}/recommendations"
    response = httpx.post(
        url,
        json=payload.model_dump(mode="json"),
        headers=_headers(),
        timeout=BACKEND_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def push_signals(currency_code: str, signals: list[AiSignalCreateRequest]) -> list[dict]:
    # AiSignalCreateRequest 엔드포인트는 signal 1개씩만 받는다 (배열 아님, openapi.yaml 기준).
    url = f"{BACKEND_BASE_URL}/internal/ai/currencies/{currency_code}/signals"
    results = []
    for signal in signals:
        response = httpx.post(
            url,
            json=signal.model_dump(mode="json"),
            headers=_headers(),
            timeout=BACKEND_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results.append(response.json())
    return results


def push_backtest_result(currency_code: str, payload: BacktestCreateRequest) -> dict:
    url = f"{BACKEND_BASE_URL}/internal/ai/currencies/{currency_code}/backtests"
    response = httpx.post(
        url,
        json=payload.model_dump(mode="json"),
        headers=_headers(),
        timeout=BACKEND_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def push_branch_recommendation(payload: BranchRecommendationPushRequest) -> dict:
    url = f"{BACKEND_BASE_URL}/internal/ai/recommendations/branches"
    response = httpx.post(
        url,
        json=payload.model_dump(mode="json"),
        headers=_headers(),
        timeout=BACKEND_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
