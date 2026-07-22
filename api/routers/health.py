from fastapi import APIRouter

from api.model_registry import list_model_versions
from api.schemas import HealthResponse, ModelStatus

router = APIRouter(prefix="/internal/ai", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="서비스 상태 및 등록된 모델 버전 조회",
)
def health() -> HealthResponse:
    """model/artifacts/manifest.json에 등록된, 전략 타입 x 통화(scope)별 "현재 서빙 버전" 목록을 반환한다.
    실제 추론은 수행하지 않는 단순 조회다."""
    versions = list_model_versions()
    return HealthResponse(status="ok", models=[ModelStatus(**v) for v in versions])
