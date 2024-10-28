FROM python:3.12-bookworm

ENV PYTHONBUFFERED True

ENV APP_HOME /HelloWorldBackend
WORKDIR $APP_HOME
COPY app.py ./


COPY requirements.txt ./

COPY . .
# Install dependencies, including a lightweight version of LaTeX

# Install LaTeX and additional tools, including cm-super, dvipng, and AWS CLI
RUN apt-get update && \
    apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-plain-generic \
    texlive-latex-extra \
    texlive-lang-cyrillic \
    cm-super \
    dvipng \
    awscli && \
    rm -rf /var/lib/apt/lists/*


EXPOSE 8080

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

ARG AWS_ACCESS_KEY_ID=AKIAUPMYM7E7VY6LNZ62
ARG AWS_SECRET_ACCESS_KEY=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

#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "1000"]
