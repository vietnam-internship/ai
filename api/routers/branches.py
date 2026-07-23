from fastapi import APIRouter

from api.schemas import BranchRecommendationRequest, BranchRecommendationResponse, ErrorResponse
from api.services.branches import build_branch_recommendations

router = APIRouter(prefix="/internal/ai", tags=["Branches"])


@router.post(
    "/branches/recommend",
    response_model=BranchRecommendationResponse,
    summary="환전소(지점) 추천 - 거리/환율/재고/예약 스코어링",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "INSUFFICIENT_DATA — 통화를 찾을 수 없거나, 반경 내 재고/예약 가능 지점이 없음",
        },
    },
)
def recommend_branches(request: BranchRecommendationRequest) -> BranchRecommendationResponse:
    """반경 내 지점 후보를 스코어링해 상위 topN개를 반환한다. 가중치는 api.routers.train의
    LOGISTIC_REGRESSION 학습 결과가 등록돼 있으면 그걸 쓰고, 없으면 Phase 1 고정 가중치로 폴백한다
    (응답의 weightsSource로 어느 쪽이 쓰였는지 확인 가능)."""
    return build_branch_recommendations(request)
