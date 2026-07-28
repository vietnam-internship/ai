from fastapi import APIRouter, BackgroundTasks

from api.schemas import BranchRecommendationRequest
from api.services.branches import run_branch_recommendation

router = APIRouter(prefix="/internal/ai", tags=["Branches"])


@router.post(
    "/branches/recommendations",
    status_code=202,
    summary="환전소(지점) 추천 비동기 연산 시작 (백엔드가 세션 생성 직후 호출)",
)
def recommend_branches(request: BranchRecommendationRequest, background_tasks: BackgroundTasks) -> dict:
    """백엔드의 POST /branches/recommendations가 PENDING 세션을 만든 뒤 호출하는 콜백.
    즉시 202로 응답하고, 실제 랭킹 연산(거리/환율/재고/예약 스코어링)은 백그라운드로 수행한 뒤
    결과를 POST /internal/ai/recommendations/branches로 push한다."""
    background_tasks.add_task(run_branch_recommendation, request)
    return {"sessionId": request.sessionId, "status": "ACCEPTED"}
