import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(REPO_ROOT / ".env")

MODEL_ARTIFACT_DIR = Path(
    os.environ.get("MODEL_ARTIFACT_DIR", str(REPO_ROOT / "model" / "artifacts"))
)

BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "http://localhost:8080")

# 접근 제어 방식이 아직 협의되지 않음 (infra 이슈 체크리스트) - 설정돼 있으면 헤더로 실어보내고,
# 없으면 그냥 호출한다. 협의 결과가 나오면 이 부분을 교체한다.
INTERNAL_AI_TOKEN = os.environ.get("INTERNAL_AI_TOKEN")

BACKEND_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("BACKEND_REQUEST_TIMEOUT_SECONDS", "10"))

DEFAULT_RECOMMENDATION_TTL_HOURS = int(os.environ.get("RECOMMENDATION_TTL_HOURS", "24"))
