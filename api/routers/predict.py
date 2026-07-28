from fastapi import APIRouter

from api.schemas import BackendPushResult, ErrorResponse, PredictAndPushResponse
from api.services.backend_client import push_recommendation, push_signals
from api.services.timing import build_timing_recommendation

router = APIRouter(prefix="/internal/ai", tags=["Predict"])


@router.post(
    "/currencies/{code}/predict",
    response_model=PredictAndPushResponse,
    summary="Compute timing recommendation and push it to the backend",
    responses={
        422: {"model": ErrorResponse, "description": "INSUFFICIENT_DATA — no exchange rate history, or below the minimum window/horizon data"},
        503: {"model": ErrorResponse, "description": "MODEL_UNAVAILABLE — unexpected failure during inference"},
    },
)
def predict_and_push(code: str) -> PredictAndPushResponse:
    """Computes the recommendation (direction/rationale/confidence) and signals (moving
    average/std) from the exchange rate history of currency `code`, then forwards them as-is
    to the backend's AI Internal push endpoints (POST .../recommendations, POST .../signals).
    Automatically falls back to baseline (30-day moving average) serving if no LR model is
    registered."""
    recommendation, signals = build_timing_recommendation(code)
    recommendation_result = push_recommendation(code, recommendation)
    signal_results = push_signals(code, signals)
    return PredictAndPushResponse(
        currencyCode=code,
        recommendation=recommendation,
        signals=signals,
        backend=BackendPushResult(recommendation=recommendation_result, signals=signal_results),
    )
