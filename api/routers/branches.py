from fastapi import APIRouter, BackgroundTasks

from api.schemas import BranchRecommendationRequest
from api.services.branches import run_branch_recommendation

router = APIRouter(tags=["Branches"])


@router.post(
    "/recommend/branches",
    status_code=202,
    summary="Start async branch recommendation computation (called by the backend right after session creation)",
)
def recommend_branches(request: BranchRecommendationRequest, background_tasks: BackgroundTasks) -> dict:
    """Callback invoked by the backend's POST /branches/recommendations after it creates a
    PENDING session (server's AiRecommendationClient.BRANCH_RANKING_PATH). Responds with 202
    immediately, runs the actual ranking computation (distance/rate/availability/reservation
    scoring) in the background, then pushes the result to POST /internal/ai/recommendations/branches."""
    background_tasks.add_task(run_branch_recommendation, request)
    return {"sessionId": request.sessionId, "status": "ACCEPTED"}
