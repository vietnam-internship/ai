from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class StrategyType(str, Enum):
    LINEAR_REGRESSION = "LINEAR_REGRESSION"
    BASE_LINE = "BASE_LINE"
    LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"


class RecommendationEnum(str, Enum):
    INCREASING = "INCREASING"
    NEUTRAL = "NEUTRAL"
    DECREASING = "DECREASING"


class SignalTypeEnum(str, Enum):
    MA = "MA"
    RSI = "RSI"
    MACD = "MACD"
    STD = "STD"


class AiSignalCreateRequest(BaseModel):
    """openapi.yaml의 AiSignalCreateRequest와 1:1 대응 (그대로 백엔드에 push됨)."""

    signalType: SignalTypeEnum
    windowDays: int
    value: float


class AiRecommendationCreateRequest(BaseModel):
    """openapi.yaml의 AiRecommendationCreateRequest와 1:1 대응 (그대로 백엔드에 push됨)."""

    signalIds: list[int] = Field(default_factory=list)
    recommendation: RecommendationEnum
    rationale: str
    confidenceScore: float
    modelVersion: str
    expiresAt: Optional[datetime] = None


class BacktestCreateRequest(BaseModel):
    """백엔드 openapi.yaml의 BacktestCreateRequest와 1:1 대응 (그대로 백엔드에 push됨)."""

    currencyId: int
    strategyType: StrategyType
    periodStart: date
    periodEnd: date
    totalSignals: int
    correctSignals: int
    accuracyRate: float


class TrainRequest(BaseModel):
    strategyType: StrategyType
    currencyCode: Optional[str] = None


class TrainResult(BaseModel):
    strategyType: StrategyType
    scope: Optional[str] = None
    modelVersion: Optional[str] = None
    source: Optional[str] = None
    metrics: Optional[dict] = None


class ModelStatus(BaseModel):
    strategyType: str
    scope: str
    version: str
    updatedAt: str


class HealthResponse(BaseModel):
    status: str
    models: list[ModelStatus]


class ErrorResponse(BaseModel):
    """api.errors의 커스텀 예외를 main.py 핸들러가 이 형태로 직렬화한다."""

    result: str = Field(examples=["FAIL"])
    code: str = Field(examples=["INSUFFICIENT_DATA", "MODEL_UNAVAILABLE"])
    message: str


class BackendPushResult(BaseModel):
    """predict가 백엔드 push 엔드포인트로부터 받은 응답을 그대로 echo한 것 (디버깅용)."""

    recommendation: dict
    signals: list[dict]


class PredictAndPushResponse(BaseModel):
    currencyCode: str
    recommendation: AiRecommendationCreateRequest
    signals: list[AiSignalCreateRequest]
    backend: BackendPushResult


class BranchRecommendationRequest(BaseModel):
    """백엔드가 POST /branches/recommendations 접수 직후 AI에 비동기로 보내는 연산 요청.
    sessionId로 결과 콜백(POST /internal/ai/recommendations/branches)을 매칭한다."""

    sessionId: int
    latitude: float
    longitude: float
    radiusKm: float
    currency: str
    amount: float


class ScoreBreakdownPayload(BaseModel):
    """RECOMMAND.feature.heuristic.score_candidates가 계산하는 개별 항목 점수 (PRD §18 w1~w4)."""

    distanceScore: float
    rateScore: float
    availabilityScore: float
    reservationScore: float


class RankedBranchItem(BaseModel):
    branchId: int
    ranking: int
    score: float
    breakdown: ScoreBreakdownPayload


class BranchRecommendationPushRequest(BaseModel):
    """백엔드 openapi.yaml의 BranchRecommendationPushRequest와 1:1 대응 (그대로 백엔드에 push됨)."""

    sessionId: int
    rankedBranches: list[RankedBranchItem]
