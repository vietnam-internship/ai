"""환전소(지점) 추천 - 규칙 기반 스코어링을 API 응답 스키마로 변환한다.

Phase 2(클릭/전환 로그 기반 로지스틱 리그레션 가중치 학습, api.routers.train의 LOGISTIC_REGRESSION
분기)로 학습된 가중치가 model_registry에 등록돼 있으면 그걸 쓰고, 없으면(로그가 아직 부족하면)
RECOMMAND.feature.heuristic.DEFAULT_WEIGHTS 고정값으로 자동 폴백한다 (services.timing의
LR-or-baseline 폴백과 동일한 패턴).
"""
from api.errors import InsufficientDataError
from api.model_registry import load_branch_weights
from api.schemas import (
    BranchRecommendationRequest,
    BranchRecommendationResponse,
    BranchScoreBreakdown,
    BranchSummary,
)


def build_branch_recommendations(request: BranchRecommendationRequest) -> BranchRecommendationResponse:
    from data_preprocessing.data_fetch import fetch_currency
    from RECOMMAND.feature.data_fetch import fetch_branch_candidates, fetch_branch_operating_hours
    from RECOMMAND.feature.heuristic import DEFAULT_RADIUS_KM, DEFAULT_TOP_N, score_candidates

    currencies = fetch_currency()
    match = currencies[currencies["code"] == request.currencyCode]
    if match.empty:
        raise InsufficientDataError(f"{request.currencyCode} 통화를 찾을 수 없습니다.")
    currency_id = int(match.iloc[0]["id"])

    candidates = fetch_branch_candidates(
        currency_id=currency_id,
        slot_date=str(request.slotDate),
        slot_time=str(request.slotTime),
    )
    if candidates.empty:
        raise InsufficientDataError(
            f"{request.currencyCode} 재고와 예약 가능 슬롯이 남아있는 지점이 없습니다."
        )

    operating_hours = fetch_branch_operating_hours(candidates["branch_id"].tolist())

    learned_weights = load_branch_weights(scope="global")
    weights_source = "logistic_regression" if learned_weights is not None else "default"

    scored = score_candidates(
        candidates,
        operating_hours,
        user_lat=request.userLat,
        user_lng=request.userLng,
        is_buying=request.isBuying,
        weights=learned_weights,
        radius_km=request.radiusKm or DEFAULT_RADIUS_KM,
        top_n=request.topN or DEFAULT_TOP_N,
    )
    if scored.empty:
        raise InsufficientDataError(
            f"반경 {request.radiusKm or DEFAULT_RADIUS_KM}km 내에 재고가 남아있는 지점이 없습니다."
        )

    branches = [
        BranchSummary(
            branchId=int(row["branch_id"]),
            branchName=row["branch_name"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            isOpenNow=bool(row["is_open_now"]),
            score=float(row["score"]),
            scoreBreakdown=BranchScoreBreakdown(
                distanceScore=float(row["distance_score"]),
                rateScore=float(row["rate_score"]),
                availabilityScore=float(row["availability_score"]),
                reservationScore=float(row["reservation_score"]),
            ),
        )
        for _, row in scored.iterrows()
    ]

    return BranchRecommendationResponse(
        currencyCode=request.currencyCode,
        weightsSource=weights_source,
        branches=branches,
    )
