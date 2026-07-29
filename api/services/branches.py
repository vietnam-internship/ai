"""환전소(지점) 추천 - 규칙 기반 스코어링을 계산해 백엔드에 콜백으로 push한다.

백엔드가 POST /branches/recommendations를 받으면 PENDING 세션을 만들고 AI에 비동기로
연산을 요청한다 (sessionId 포함). 여기서는 그 연산을 수행하고 결과를
POST /internal/ai/recommendations/branches로 push한다 (동기 응답이 아님 - 호출부가
FastAPI BackgroundTasks로 백그라운드 실행한다).

Phase 2(클릭/전환 로그 기반 로지스틱 리그레션 가중치 학습, api.routers.train의 LOGISTIC_REGRESSION
분기)로 학습된 가중치가 model_registry에 등록돼 있으면 그걸 쓰고, 없으면(로그가 아직 부족하면)
RECOMMAND.feature.heuristic.DEFAULT_WEIGHTS 고정값으로 자동 폴백한다 (services.timing의
LR-or-baseline 폴백과 동일한 패턴).
"""
import logging
import time
from datetime import date, datetime

from api.model_registry import load_branch_weights

logger = logging.getLogger(__name__)
from api.schemas import (
    BranchRecommendationPushRequest,
    BranchRecommendationRequest,
    RankedBranchItem,
    ScoreBreakdownPayload,
)
from api.services.backend_client import push_branch_recommendation

_SLOT_MINUTES = 30


def _current_slot(now: datetime) -> tuple[date, str]:
    floored_minute = (now.minute // _SLOT_MINUTES) * _SLOT_MINUTES
    slot_time = now.replace(minute=floored_minute, second=0, microsecond=0)
    return slot_time.date(), slot_time.strftime("%H:%M:%S")


def run_branch_recommendation(request: BranchRecommendationRequest) -> None:
    from RECOMMAND.feature.data_fetch import fetch_branch_candidates, fetch_branch_operating_hours
    from RECOMMAND.feature.heuristic import score_candidates

    t0 = time.time()
    logger.info("[추천 시작] sessionId=%s, currency=%s, lat=%s, lng=%s, radiusKm=%s",
                request.sessionId, request.currencyCode, request.latitude, request.longitude, request.radiusKm)

    try:
        slot_date, slot_time = _current_slot(datetime.now())
        candidates = fetch_branch_candidates(request.currencyCode, str(slot_date), slot_time)
        logger.info("[DB 조회 완료] sessionId=%s, 후보 수=%d, 소요=%.2fs",
                    request.sessionId, len(candidates), time.time() - t0)

        if candidates.empty:
            logger.warning("[후보 없음] sessionId=%s, currency=%s, slot=%s %s — 콜백 생략, PENDING 유지",
                           request.sessionId, request.currencyCode, slot_date, slot_time)
            return

        operating_hours = fetch_branch_operating_hours(candidates["branch_id"].tolist())
        learned_weights = load_branch_weights(scope="global")

        scored = score_candidates(
            candidates,
            operating_hours,
            user_lat=request.latitude,
            user_lng=request.longitude,
            is_buying=True,
            weights=learned_weights,
            radius_km=request.radiusKm,
        )
        logger.info("[스코어링 완료] sessionId=%s, 반경 내 지점=%d, 소요=%.2fs",
                    request.sessionId, len(scored), time.time() - t0)

        if scored.empty:
            logger.warning("[반경 내 후보 없음] sessionId=%s, radiusKm=%s — 콜백 생략",
                           request.sessionId, request.radiusKm)
            return

        ranked_items = [
            RankedBranchItem(
                branchId=int(row["branch_id"]),
                ranking=idx + 1,
                score=float(row["score"]),
                breakdown=ScoreBreakdownPayload(
                    distanceScore=float(row["distance_score"]),
                    rateScore=float(row["rate_score"]),
                    availabilityScore=float(row["availability_score"]),
                    reservationScore=float(row["reservation_score"]),
                ),
            )
            for idx, row in scored.reset_index(drop=True).iterrows()
        ]

        push_branch_recommendation(
            BranchRecommendationPushRequest(sessionId=request.sessionId, rankedBranches=ranked_items)
        )
        logger.info("[콜백 push 완료] sessionId=%s, 총 소요=%.2fs", request.sessionId, time.time() - t0)

    except Exception as e:
        logger.error("[추천 실패] sessionId=%s, 총 소요=%.2fs, error=%s",
                     request.sessionId, time.time() - t0, e, exc_info=True)
