FROM python:3.8.12-buster

COPY taxifare /taxifare
COPY requirements_prod.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
