FROM python:3.10-slim

COPY . /django-ollama-rag
WORKDIR /django-ollama-rag

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN uv python list

RUN --mount=type=cache,target=/tmp/uv-cache \
    uv pip install --no-cache-dir -r requirements.txt --system


CMD python manage.py wait_for_db \
    && python manage.py makemigrations \
    && python manage.py migrate \
    && python manage.py runserver 0.0.0.0:8000