from fastapi import APIRouter

from api.errors import InsufficientDataError
from api.model_registry import save_lr_model
from api.schemas import ErrorResponse, StrategyType, TrainRequest, TrainResult

router = APIRouter(prefix="/internal/ai", tags=["Train"])


@router.post(
    "/train",
    response_model=TrainResult,
    summary="전략 타입별 모델 학습/조회",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "INSUFFICIENT_DATA — currencyCode 누락, 환율 이력 없음, 또는 LOGISTIC_REGRESSION 미지원",
        },
    },
)
def train(request: TrainRequest) -> TrainResult:
    """strategyType에 따라 동작이 다르다.

    - LINEAR_REGRESSION: currencyCode 필수. LR 모델을 학습하고 baseline보다 우수한 경우에만
      .joblib로 저장 + manifest 갱신(source: "lr"). 그렇지 않으면 아티팩트 없이 스킵 사유만 반환.
    - BASE_LINE: currencyCode 필수. 30일 이동평균 규칙의 MAE/MAPE만 계산해 반환 (저장할 아티팩트 없음).
    - LOGISTIC_REGRESSION: 클릭/전환 로그가 아직 없어 학습 불가. 항상 422 INSUFFICIENT_DATA."""
    if request.strategyType == StrategyType.LINEAR_REGRESSION:
        return _train_lr(request.currencyCode)
    if request.strategyType == StrategyType.BASE_LINE:
        return _train_baseline(request.currencyCode)
    # LOGISTIC_REGRESSION: 클릭/전환 로그가 아직 없어 실학습 불가 (infra 이슈 blocker).
    # Phase 1 룰 기반 고정 가중치를 그대로 쓰도록 안내한다.
    raise InsufficientDataError(
        "클릭/전환 로그 데이터가 아직 없어 Logistic Regression을 학습할 수 없습니다. "
        "Phase 1 룰 기반 가중치를 계속 사용하세요."
    )


def _train_lr(currency_code: str | None) -> TrainResult:
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
    from model.lr_model import build_and_train_with_fallback

    if not currency_code:
        raise InsufficientDataError("LINEAR_REGRESSION 학습에는 currencyCode가 필요합니다.")

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=currency_code)
    if df.empty:
        raise InsufficientDataError(f"{currency_code} 환율 데이터가 없습니다.")

    result = build_and_train_with_fallback(df=df)

    if result["source"] != "lr":
        # 데이터 부족 또는 LR이 baseline보다 못해서 학습을 스킵한 경우. 아티팩트는 저장하지 않는다.
        return TrainResult(
            strategyType=StrategyType.LINEAR_REGRESSION,
            scope=currency_code,
            source=result["source"],
            metrics={
                "reason": result["reason"],
                "lrMae": result["lr_mae"],
                "baselineMae": result["baseline_mae"],
            },
        )

    entry = save_lr_model(result["model"], currency_code)
    return TrainResult(
        strategyType=StrategyType.LINEAR_REGRESSION,
        scope=currency_code,
        modelVersion=entry["version"],
        source="lr",
        metrics={"lrMae": result["lr_mae"], "baselineMae": result["baseline_mae"]},
    )


def _train_baseline(currency_code: str | None) -> TrainResult:
    from baseline import metrics as baseline_metrics
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries

    if not currency_code:
        raise InsufficientDataError("BASE_LINE 조회에는 currencyCode가 필요합니다.")

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=currency_code)
    try:
        mae = baseline_metrics.mae(df)
        mape = baseline_metrics.mape(df)
    except ValueError as exc:
        raise InsufficientDataError(str(exc)) from exc

    # baseline은 학습 파라미터가 없는 규칙(30일 이동평균)이라 저장할 아티팩트가 없다.
    return TrainResult(
        strategyType=StrategyType.BASE_LINE,
        scope=currency_code,
        source="baseline",
        metrics={"mae": mae, "mape": mape},
    )
