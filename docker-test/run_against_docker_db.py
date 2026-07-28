"""docker-test/docker-compose.yml로 띄운 진짜 MySQL을 상대로 /train, /predict를 돌려보는 스크립트.

mock_run.py와 다른 점: 환율 데이터는 진짜 DB(SELECT)에서 읽어온다.
다만 실제 백엔드(Spring) 서버까지 띄우는 건 범위 밖이라, predict의 push 부분만 그대로 가짜로 남겨둔다.

사용법:
    1) docker compose -f docker-test/docker-compose.yml up -d   (ai/ 에서 실행)
    2) DB가 준비될 때까지 몇 초 대기 (헬스체크 통과 확인: docker compose -f docker-test/docker-compose.yml ps)
    3) python3 docker-test/run_against_docker_db.py             (역시 ai/ 에서 실행)
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
AI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(AI_ROOT))  # ai/ 를 import 루트로 추가

from dotenv import load_dotenv

# data_fetch.py가 나중에 load_dotenv()를 다시 호출해도, 이미 로드된 값은 override하지 않으므로
# 여기서 먼저 이 폴더의 .env(테스트 DB 접속 정보)를 명시적으로 로드해둔다.
load_dotenv(THIS_DIR / ".env")

from fastapi.testclient import TestClient

from api.main import app
import api.routers.predict as predict_router_module

# 실제 백엔드(Spring) 서버는 띄우지 않으므로, push만 가짜로 응답하게 한다.
predict_router_module.push_recommendation = lambda code, payload: {"id": 1, "mock": True}
predict_router_module.push_signals = lambda code, signals: [
    {"id": i + 1, "mock": True} for i in range(len(signals))
]

client = TestClient(app)


def show(title: str, response) -> None:
    print(f"\n=== {title} ===")
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    show("health (학습 전)", client.get("/internal/ai/health"))

    show(
        "train: BASE_LINE",
        client.post("/internal/ai/train", json={"strategyType": "BASE_LINE", "currencyCode": "USD"}),
    )

    show(
        "train: LINEAR_REGRESSION",
        client.post("/internal/ai/train", json={"strategyType": "LINEAR_REGRESSION", "currencyCode": "USD"}),
    )

    show("health (학습 후)", client.get("/internal/ai/health"))

    show("predict", client.post("/internal/ai/currencies/USD/predict"))
