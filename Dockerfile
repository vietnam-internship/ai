FROM python:3.12-slim

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
COPY data_preprocessing/requirements.txt data_preprocessing/requirements.txt
COPY model/requirements.txt model/requirements.txt

RUN pip install --no-cache-dir -r api/requirements.txt

COPY api api
COPY data_preprocessing data_preprocessing
COPY model model
COPY baseline baseline
COPY RECOMMAND RECOMMAND

ENV MODEL_ARTIFACT_DIR=/app/model/artifacts

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
