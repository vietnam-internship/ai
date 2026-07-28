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
        raise InsufficientDataError(f"No currency info found for {currency_code}.")
    return int(match.iloc[0]["id"])


@router.post(
    "/currencies/{code}/backtest",
    summary="Run LR model walk-forward backtest and push the result to the backend",
)
def backtest(code: str) -> dict:
    """Runs walk-forward validation of the LR model against actual exchange rate history in
    the DB, converts direction (sign) agreement into signal-level accuracy
    (totalSignals/correctSignals/accuracyRate), and pushes it to the backend's
    POST /internal/ai/currencies/{code}/backtests."""
    from data_preprocessing.preprocess import fetch_and_fill_exchange_rate_timeseries
    from model.backtest import backtest_currency

    df = fetch_and_fill_exchange_rate_timeseries(cur_unit=code)
    if df.empty:
        raise InsufficientDataError(f"No exchange rate data for {code}.")

    try:
        result = backtest_currency(df)
    except ValueError as exc:
        raise InsufficientDataError(str(exc)) from exc

    if not result["totalSignals"]:
        raise InsufficientDataError(f"No backtest signals for {code}.")

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
