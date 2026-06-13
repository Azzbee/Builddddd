.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help up down logs ps build seed ingest test eval lint typecheck fmt check db-schema web-dev api-dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Bring up the full stack (neo4j, postgres, redis, grobid, api, worker, web)
	$(COMPOSE) up -d --build
	@echo "API:   http://localhost:8000/health"
	@echo "Web:   http://localhost:3000"
	@echo "Neo4j: http://localhost:7474"

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail all logs
	$(COMPOSE) logs -f

ps: ## Show service status
	$(COMPOSE) ps

build: ## Build images
	$(COMPOSE) build

db-schema: ## Apply the Postgres schema
	$(COMPOSE) exec -T postgres psql -U lattice -d lattice < backend/lattice/db/schema.sql

seed: ## Pull ~20 arXiv commodity-forecasting papers into the corpus
	cd backend && uv run python ../scripts/seed_corpus.py

ingest: ## Ingest a local PDF: make ingest FILE=path/to/paper.pdf
	@test -n "$(FILE)" || (echo "usage: make ingest FILE=paper.pdf" && exit 1)
	curl -sS -F "file=@$(FILE)" http://localhost:8000/ingest/file | python3 -m json.tool

test: ## Run the backend test suite
	cd backend && uv run pytest

eval: ## Run extraction + retrieval + edge-quality evals
	cd backend && uv run python ../scripts/run_eval.py

lint: ## Ruff lint (backend) + eslint (web)
	cd backend && uv run ruff check .
	cd web && npm run lint --if-present

typecheck: ## mypy (backend) + tsc (web)
	cd backend && uv run mypy lattice
	cd web && npm run typecheck --if-present

fmt: ## Auto-format / fix
	cd backend && uv run ruff check --fix . && uv run ruff format .

check: lint typecheck test ## Run lint + typecheck + tests

api-dev: ## Run the API locally with reload
	cd backend && uv run uvicorn lattice.api.app:app --reload

demo: ## Run the API in offline demo mode (populated graph, no external services)
	cd backend && LATTICE_DEMO_MODE=true uv run uvicorn lattice.api.app:app --reload
	@echo "Open the web app (make web-dev) against this API to explore the demo graph."

web-dev: ## Run the Next.js dev server
	cd web && npm run dev
