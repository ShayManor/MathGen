FROM python:3.12-bookworm

ENV PYTHONBUFFERED True

ENV APP_HOME /HelloWorldBackend
WORKDIR $APP_HOME
COPY app.py ./


COPY requirements.txt ./

COPY . .
EXPOSE 8080

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]