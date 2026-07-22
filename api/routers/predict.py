from fastapi import APIRouter

from api.schemas import BackendPushResult, ErrorResponse, PredictAndPushResponse
from api.services.backend_client import push_recommendation, push_signals
from api.services.timing import build_timing_recommendation

router = APIRouter(prefix="/internal/ai", tags=["Predict"])


@router.post(
    "/currencies/{code}/predict",
    response_model=PredictAndPushResponse,
    summary="타이밍 추천 계산 후 백엔드에 push",
    responses={
        422: {"model": ErrorResponse, "description": "INSUFFICIENT_DATA — 환율 이력 없음 또는 최소 window/horizon 데이터 부족"},
        503: {"model": ErrorResponse, "description": "MODEL_UNAVAILABLE — 추론 중 예기치 못한 실패"},
    },
)
def predict_and_push(code: str) -> PredictAndPushResponse:
    """code 통화의 환율 이력으로 추천(방향/근거/신뢰도)과 시그널(이동평균/표준편차)을 계산한 뒤,
    백엔드의 AI Internal push 엔드포인트(POST .../recommendations, POST .../signals)로 그대로 전달한다.
    등록된 LR 모델이 없으면 baseline(30일 이동평균)으로 자동 폴백해서 서빙한다."""
    recommendation, signals = build_timing_recommendation(code)
    recommendation_result = push_recommendation(code, recommendation)
    signal_results = push_signals(code, signals)
    return PredictAndPushResponse(
        currencyCode=code,
        recommendation=recommendation,
        signals=signals,
        backend=BackendPushResult(recommendation=recommendation_result, signals=signal_results),
    )
