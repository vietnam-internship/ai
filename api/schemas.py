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
    """Matches AiSignalCreateRequest in openapi.yaml 1:1 (pushed to the backend as-is)."""

    signalType: SignalTypeEnum
    windowDays: int
    value: float


class AiRecommendationCreateRequest(BaseModel):
    """Matches AiRecommendationCreateRequest in openapi.yaml 1:1 (pushed to the backend as-is)."""

    signalIds: list[int] = Field(default_factory=list)
    recommendation: RecommendationEnum
    rationale: str
    confidenceScore: float
    modelVersion: str
    expiresAt: Optional[datetime] = None


class BacktestCreateRequest(BaseModel):
    """Matches BacktestCreateRequest in the backend's openapi.yaml 1:1 (pushed to the backend as-is)."""

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
    """The shape main.py's handlers serialize api.errors custom exceptions into."""

    result: str = Field(examples=["FAIL"])
    code: str = Field(examples=["INSUFFICIENT_DATA", "MODEL_UNAVAILABLE"])
    message: str


class BackendPushResult(BaseModel):
    """Echoes back the response predict got from the backend push endpoints (for debugging)."""

    recommendation: dict
    signals: list[dict]


class PredictAndPushResponse(BaseModel):
    currencyCode: str
    recommendation: AiRecommendationCreateRequest
    signals: list[AiSignalCreateRequest]
    backend: BackendPushResult


class BranchRecommendationRequest(BaseModel):
    """The async computation request the backend sends to AI right after accepting
    POST /branches/recommendations. sessionId matches it to the result callback
    (POST /internal/ai/recommendations/branches)."""

    sessionId: int
    latitude: float
    longitude: float
    radiusKm: float
    currencyCode: str
    amount: float


class ScoreBreakdownPayload(BaseModel):
    """Per-item scores computed by RECOMMAND.feature.heuristic.score_candidates (PRD §18 w1~w4)."""

    distanceScore: float
    rateScore: float
    availabilityScore: float
    reservationScore: float


class RankedBranchItem(BaseModel):
    branchId: int
    ranking: int
    score: float
    breakdown: Optional[ScoreBreakdownPayload] = None


class BranchRecommendationPushRequest(BaseModel):
    """Matches BranchRecommendationPushRequest in the backend's openapi.yaml 1:1 (pushed to the backend as-is)."""

    sessionId: int
    rankedBranches: list[RankedBranchItem]
