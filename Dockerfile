  FROM python:3.10.10-bullseye
  COPY LeWaGAN LeWaGAN
  COPY requirements.txt requirements.txt
  COPY local_training/model_saves local_training/model_saves
  RUN pip install --upgrade pip
  RUN pip install -r requirements.txt
  CMD uvicorn LeWaGAN.api.fast:app --host 0.0.0.0 --port $PORT
