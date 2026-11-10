# One image for every app service (api, worker, ui), so they all run the same
# scikit-learn version that serialized pipelines depend on.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_PATH=/srv/data/training_data.csv

WORKDIR /srv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY data ./data
RUN uv sync --frozen --no-dev

# The commit the image was built from, recorded on every Run (.git is not in the image).
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

CMD ["python", "-c", "import app.settings; print('experimentation-platform image')"]
