from fastapi import APIRouter

from api.model_registry import list_model_versions
from api.schemas import HealthResponse, ModelStatus

router = APIRouter(prefix="/internal/ai", tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service status and registered model versions",
)
def health() -> HealthResponse:
    """Returns the list of "currently serving version" per strategy type x currency (scope)
    registered in model/artifacts/manifest.json. A plain lookup that runs no actual inference."""
    versions = list_model_versions()
    return HealthResponse(status="ok", models=[ModelStatus(**v) for v in versions])
