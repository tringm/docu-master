FROM python:3.10.13

WORKDIR /app

RUN pip install --upgrade pip && pip install poetry==1.7.1

COPY pyproject.toml .
COPY poetry.toml .
COPY poetry.lock .

RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-root

COPY src ./src

CMD ["python", "-m", "src.app"]
