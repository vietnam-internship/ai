"""학습된 모델(.joblib) 아티팩트 저장/로드와 "현재 서빙 버전" 매니페스트 관리.

infra 이슈에서 정한 네이밍: {prefix}_{scope}_{YYYYMMDD}.joblib
"현재 서빙 버전"은 latest 심볼릭 링크 대신 manifest.json으로 관리한다 (여러 scope를
독립적으로 롤백해야 해서 파일 하나짜리 포인터 구조가 더 다루기 쉽다).
"""
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from api.config import MODEL_ARTIFACT_DIR

MANIFEST_PATH = MODEL_ARTIFACT_DIR / "manifest.json"


def _manifest_key(strategy_type: str, scope: str) -> str:
    return f"{strategy_type}:{scope}"


def _read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def _write_manifest(manifest: dict) -> None:
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def artifact_filename(prefix: str, scope: str, on: Optional[date] = None, ext: str = "joblib") -> str:
    on = on or date.today()
    return f"{prefix}_{scope}_{on.strftime('%Y%m%d')}.{ext}"


def save_lr_model(model, scope: str) -> dict:
    """LR 모델을 .joblib로 저장하고 manifest의 LINEAR_REGRESSION:{scope} 항목을 갱신한다.
    이전 아티팩트 파일은 지우지 않는다 (롤백 시 필요)."""
    from model.inference import save_model

    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    filename = artifact_filename("lr", scope)
    path = MODEL_ARTIFACT_DIR / filename
    save_model(model, str(path))

    entry = {
        "version": filename,
        "path": str(path),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    manifest = _read_manifest()
    manifest[_manifest_key("LINEAR_REGRESSION", scope)] = entry
    _write_manifest(manifest)
    return entry


def load_lr_model(scope: str):
    """등록된 LR 아티팩트가 있으면 (model, version)을, 없으면 (None, None)을 반환한다.
    모델이 없을 때 baseline으로 대체하는 것은 호출부(services.timing)의 책임이다."""
    from model.inference import load_model

    entry = _read_manifest().get(_manifest_key("LINEAR_REGRESSION", scope))
    if entry is None:
        return None, None
    return load_model(entry["path"]), entry["version"]


def save_branch_weights(weights: dict, scope: str = "global") -> dict:
    """로지스틱 리그레션으로 학습한 지점 추천 가중치(w1~w4)를 .json으로 저장하고
    manifest의 LOGISTIC_REGRESSION:{scope} 항목을 갱신한다. LR(.joblib)과 달리 통화별이 아니라
    전역(global) 가중치라 scope 기본값을 "global"로 둔다."""
    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    filename = artifact_filename("blr", scope, ext="json")
    path = MODEL_ARTIFACT_DIR / filename
    path.write_text(json.dumps(weights, indent=2))

    entry = {
        "version": filename,
        "path": str(path),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    manifest = _read_manifest()
    manifest[_manifest_key("LOGISTIC_REGRESSION", scope)] = entry
    _write_manifest(manifest)
    return entry


def load_branch_weights(scope: str = "global") -> Optional[dict]:
    """등록된 학습 가중치가 있으면 dict({"distance":..,"rate":..,"availability":..,"reservation":..})를,
    없으면 None을 반환한다. 없을 때 DEFAULT_WEIGHTS로 대체하는 것은 호출부(services.branches)의 책임이다."""
    entry = _read_manifest().get(_manifest_key("LOGISTIC_REGRESSION", scope))
    if entry is None:
        return None
    return json.loads(Path(entry["path"]).read_text())


def list_model_versions() -> list[dict]:
    manifest = _read_manifest()
    versions = []
    for key, value in manifest.items():
        strategy_type, scope = key.split(":", 1)
        versions.append({"strategyType": strategy_type, "scope": scope, **value})
    return versions
