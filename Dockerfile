FROM python:3.12-bookworm

ENV PYTHONBUFFERED True

ENV APP_HOME /HelloWorldBackend
WORKDIR $APP_HOME
COPY app.py ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app