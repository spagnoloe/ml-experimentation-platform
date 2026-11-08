.PHONY: install lint format test db up down logs image train ui

# Recorded on every Run by the worker image.
export GIT_SHA := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

# Start the Compose Postgres (if needed) and wait until it is healthy.
db:
	docker compose up -d --wait postgres

# The suite drops and recreates app_test at the start of each session.
test: db
	uv run pytest

up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f

image:
	docker build --build-arg GIT_SHA=$(GIT_SHA) -t adp-experimentation-platform .

# Submit an Experiment Config and wait for the worker to finish it. Needs `make up`.
train:
	@test -n "$(CONFIG)" || (echo "usage: make train CONFIG=configs/rf-fixed.yaml" && exit 2)
	uv run python -m app.cli submit $(CONFIG) --wait

# Run the UI on the host against the Compose stack (`make up`), at http://localhost:8501.
ui:
	uv run streamlit run app/ui/main.py
