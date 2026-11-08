.PHONY: install lint format test db up down logs image

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
	docker compose up -d --wait

down:
	docker compose down

logs:
	docker compose logs -f

image:
	docker build -t adp-experimentation-platform .
