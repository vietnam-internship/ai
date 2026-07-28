class InsufficientDataError(Exception):
    """422 INSUFFICIENT_DATA - 예측/학습에 필요한 최소 데이터가 없을 때."""

    def __init__(self, message: str = "Not enough data."):
        self.message = message
        super().__init__(message)


class ModelUnavailableError(Exception):
    """503 MODEL_UNAVAILABLE - 모델 로딩/추론이 예기치 못하게 실패했을 때."""

    def __init__(self, message: str = "Model is unavailable."):
        self.message = message
        super().__init__(message)
