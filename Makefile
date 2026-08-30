.DEFAULT_GOAL := help
SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help up down logs ps build seed ingest test web-test eval lint typecheck fmt format-check web-build audit check db-schema web-dev api-dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up: ## Bring up the full stack (neo4j, postgres, redis, grobid, api, worker, web)
	$(COMPOSE) up -d --build
	@echo "API:   http://localhost:8000/readyz"
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

ingest: ## Ingest a local PDF: make ingest FILE=paper.pdf TOKEN=secret
	@test -n "$(FILE)" || (echo "usage: make ingest FILE=paper.pdf" && exit 1)
	curl -sS -H "Authorization: Bearer $(TOKEN)" -F "file=@$(FILE)" http://localhost:8000/ingest/file | python3 -m json.tool

test: ## Run the backend test suite (offline; integration tests skip)
	cd backend && uv run pytest

web-test: ## Run the frontend unit tests
	cd web && npm run test

test-integration: ## Run live integration tests (needs LATTICE_TEST_PG_DSN / LATTICE_TEST_NEO4J_URI)
	cd backend && uv run pytest -m integration -v

test-e2e: ## Run the live GROBID end-to-end suite (starts GROBID, adds the LLM leg if LATTICE_E2E_LLM_MODEL is set)
	-docker rm -f lattice-grobid-e2e 2>/dev/null
	docker run --rm -d --name lattice-grobid-e2e -p 8070:8070 lfoppiano/grobid:0.8.1
	@echo "waiting for GROBID..."
	@until curl -sf http://localhost:8070/api/isalive > /dev/null; do sleep 5; done
	-cd backend && LATTICE_E2E_GROBID_URL=http://localhost:8070 uv run pytest -m e2e -v
	docker rm -f lattice-grobid-e2e

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
	cd web && npm run format

format-check: ## Check backend and frontend formatting
	cd backend && uv run ruff format --check .
	cd web && npm run format:check

web-build: ## Build the web app into its verification directory
	cd web && npm run build:check

audit: ## Audit frontend production and development dependencies
	cd web && npm audit --audit-level=moderate

check: format-check lint typecheck test web-test web-build ## Run the local verification gate

api-dev: ## Run the API locally with reload
	cd backend && uv run uvicorn lattice.api.app:app --reload

demo: ## Run the API in offline demo mode (populated graph, no external services)
	cd backend && LATTICE_DEMO_MODE=true uv run uvicorn lattice.api.app:app --reload
	@echo "Open the web app (make web-dev) against this API to explore the demo graph."

web-dev: ## Run the Next.js dev server
	cd web && npm run dev
