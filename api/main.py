from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors import InsufficientDataError, ModelUnavailableError
from api.routers import backtest, branches, health, predict, train

app = FastAPI(
    title="TravelX AI Internal Service",
    version="0.1.0",
    description=(
        "Internal service that trains/serves the exchange rate timing recommendation model. "
        "The AI Internal tags documented by the backend (recommendations/signals, etc.) are "
        "the endpoints this service calls when pushing to the backend; the endpoints defined "
        "here are the API this service exposes itself."
    ),
)

app.include_router(train.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(branches.router)
app.include_router(backtest.router)


@app.exception_handler(InsufficientDataError)
async def handle_insufficient_data(request: Request, exc: InsufficientDataError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"result": "FAIL", "code": "INSUFFICIENT_DATA", "message": exc.message},
    )


@app.exception_handler(ModelUnavailableError)
async def handle_model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"result": "FAIL", "code": "MODEL_UNAVAILABLE", "message": exc.message},
    )
