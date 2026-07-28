from fastapi import APIRouter

from api.errors import InsufficientDataError
from api.schemas import BacktestCreateRequest, StrategyType
from api.services.backend_client import push_backtest_result

router = APIRouter(prefix="/internal/ai", tags=["Backtest"])


def _currency_id(currency_code: str) -> int:
    from data_preprocessing.data_fetch import fetch_currency

    currencies = fetch_currency()
    match = currencies.loc[currencies["code"] == currency_code]
    if match.empty:
        raise InsufficientDataError(f"{currency_code}에 대한 통화 정보가 없습니다.")
    return int(match.iloc[0]["id"])


@router.post(
    "/currencies/{code}/backtest",
    summary="LR 모델 walk-forward 백테스트 실행 후 백엔드에 결과 push",
)
def backtest(code: str) -> dict:
    """DB에 쌓인 실제 환율 이력으로 LR 모델을 walk-forward validation하고, 방향(부호) 일치
    여부를 신호 단위 정확도(totalSignals/correctSignals/accuracyRate)로 환산해 백엔드의
    POST /internal/ai/currencies/{code}/backtests로 push한다."""
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
    from model.backtest import backtest_currency

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=code)
    if df.empty:
        raise InsufficientDataError(f"{code} 환율 데이터가 없습니다.")

    try:
        result = backtest_currency(df)
    except ValueError as exc:
        raise InsufficientDataError(str(exc)) from exc

    if not result["totalSignals"]:
        raise InsufficientDataError(f"{code}에 대한 백테스트 신호가 없습니다.")

    payload = BacktestCreateRequest(
        currencyId=_currency_id(code),
        strategyType=StrategyType.LINEAR_REGRESSION,
        periodStart=df["date"].min(),
        periodEnd=df["date"].max(),
        totalSignals=result["totalSignals"],
        correctSignals=result["correctSignals"],
        accuracyRate=result["accuracyRate"],
    )
    backend_result = push_backtest_result(code, payload)

    return {
        "currencyCode": code,
        "backtest": payload,
        "backend": backend_result,
    }
