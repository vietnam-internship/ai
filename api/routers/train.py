from fastapi import APIRouter

from api.errors import InsufficientDataError
from api.model_registry import save_branch_weights, save_lr_model
from api.schemas import ErrorResponse, StrategyType, TrainRequest, TrainResult

MIN_BRANCH_FEEDBACK_ROWS = 100  # PRD 협의 전 임시 임계값. 로그 특성 파악되면 조정.

router = APIRouter(prefix="/internal/ai", tags=["Train"])


@router.post(
    "/train",
    response_model=TrainResult,
    summary="Train/lookup model by strategy type",
    responses={
        422: {
            "model": ErrorResponse,
            "description": (
                "INSUFFICIENT_DATA — currencyCode missing, no exchange rate history, or "
                "LOGISTIC_REGRESSION click/conversion logs below the minimum count"
            ),
        },
    },
)
def train(request: TrainRequest) -> TrainResult:
    """Behavior depends on strategyType.

    - LINEAR_REGRESSION: currencyCode required. Trains an LR model and only saves it as
      .joblib + updates the manifest (source: "lr") if it beats the baseline. Otherwise
      returns the skip reason without saving an artifact.
    - BASE_LINE: currencyCode required. Only computes and returns MAE/MAPE for the 30-day
      moving average rule (no artifact to save).
    - LOGISTIC_REGRESSION: branch recommendation Phase 2. Once click/conversion logs reach
      MIN_BRANCH_FEEDBACK_ROWS, trains w1~w4 weights via logistic regression and saves them.
      Before that, returns 422 INSUFFICIENT_DATA advising to keep using the Phase 1 fixed
      rule-based weights."""
    if request.strategyType == StrategyType.LINEAR_REGRESSION:
        return _train_lr(request.currencyCode)
    if request.strategyType == StrategyType.BASE_LINE:
        return _train_baseline(request.currencyCode)
    return _train_branch_recommendation_weights()


def _train_lr(currency_code: str | None) -> TrainResult:
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
    from model.lr_model import build_and_train_with_fallback

    if not currency_code:
        raise InsufficientDataError("currencyCode is required for LINEAR_REGRESSION training.")

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=currency_code)
    if df.empty:
        raise InsufficientDataError(f"No exchange rate data for {currency_code}.")

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
        raise InsufficientDataError("currencyCode is required for BASE_LINE lookup.")

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


def _train_branch_recommendation_weights() -> TrainResult:
    """지점 추천 Phase 2: branch_recommendation_feedback 로그로 w1~w4(거리/환율/재고/예약)
    가중치를 로지스틱 리그레션으로 학습한다. RECOMMAND.model.weight_model.build_and_train_with_fallback을
    통해 baseline(DEFAULT_WEIGHTS)보다 못하거나 계수가 음수면 학습 결과를 버리고 Phase 1 고정
    가중치를 계속 사용한다 (model.lr_model의 LR-or-baseline 폴백과 동일한 패턴)."""
    from RECOMMAND.feature.data_fetch import fetch_branch_recommendation_logs
    from RECOMMAND.model.weight_model import build_and_train_with_fallback

    logs = fetch_branch_recommendation_logs()
    if logs.empty or len(logs) < MIN_BRANCH_FEEDBACK_ROWS:
        raise InsufficientDataError(
            "Not enough click/conversion log data to train Logistic Regression "
            f"(minimum {MIN_BRANCH_FEEDBACK_ROWS} rows required). Keep using the Phase 1 rule-based weights."
        )

    result = build_and_train_with_fallback(logs=logs, min_samples=MIN_BRANCH_FEEDBACK_ROWS)

    if result["source"] != "logistic_regression":
        # 계수가 음수이거나 baseline보다 못해서 학습을 스킵한 경우. 아티팩트는 저장하지 않고
        # Phase 1 고정 가중치를 계속 쓴다.
        return TrainResult(
            strategyType=StrategyType.LOGISTIC_REGRESSION,
            scope="global",
            source=result["source"],
            metrics={"reason": result["reason"], "trainRows": result["trainRows"], **result["metrics"]},
        )

    entry = save_branch_weights(result["weights"], scope="global")
    return TrainResult(
        strategyType=StrategyType.LOGISTIC_REGRESSION,
        scope="global",
        modelVersion=entry["version"],
        source="logistic_regression",
        metrics={"weights": result["weights"], "trainRows": result["trainRows"], **result["metrics"]},
    )
