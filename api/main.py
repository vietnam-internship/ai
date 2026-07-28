from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors import InsufficientDataError, ModelUnavailableError
from api.routers import backtest, branches, health, predict, train

app = FastAPI(
    title="TravelX AI Internal Service",
    version="0.1.0",
    description=(
        "환율 타이밍 추천 모델을 학습/서빙하는 내부 전용 서비스. "
        "백엔드가 문서화한 AI Internal 태그(recommendations/signals 등)는 이 서비스가 "
        "백엔드로 push할 때 호출하는 엔드포인트이고, 여기 정의된 엔드포인트는 이 서비스 "
        "자신이 노출하는 API다."
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
