-- Lattice Postgres schema (Supabase-compatible). Every row carries workspace_id.
-- Vectors use pgvector. Apply with: psql "$DSN" -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL DEFAULT 'default',
    title           TEXT NOT NULL,
    authors         JSONB NOT NULL DEFAULT '[]',
    year            INT,
    venue           TEXT,
    doi             TEXT,
    arxiv_id        TEXT,
    s2_paper_id     TEXT,
    card            JSONB NOT NULL DEFAULT '{}',   -- full PaperCard
    specter         vector(768),
    confidence      REAL,
    needs_review    BOOLEAN NOT NULL DEFAULT FALSE,
    content_hash    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS papers_workspace ON papers (workspace_id);
CREATE INDEX IF NOT EXISTS papers_doi ON papers (doi);
CREATE INDEX IF NOT EXISTS papers_arxiv ON papers (arxiv_id);
CREATE INDEX IF NOT EXISTS papers_hash ON papers (content_hash);
CREATE INDEX IF NOT EXISTS papers_title_trgm ON papers USING gin (title gin_trgm_ops);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    paper_id          TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    workspace_id      TEXT NOT NULL DEFAULT 'default',
    title             TEXT NOT NULL,
    section_title     TEXT NOT NULL,
    text              TEXT NOT NULL,
    embedding         vector(1024),
    evidence_location TEXT,
    ordinal           INT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_paper ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS chunks_workspace ON chunks (workspace_id);
CREATE INDEX IF NOT EXISTS chunks_fts ON chunks USING gin (to_tsvector('english', text));
-- ANN index (cosine). Build after bulk load for best results.
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id        TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'default',
    source_type   TEXT NOT NULL,
    source_ref    TEXT NOT NULL,
    content_hash  TEXT,
    paper_id      TEXT,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    error_code    TEXT,
    error_message TEXT,
    attempts      INT NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status ON ingest_jobs (status);

-- Every weight change, old -> new, with reason. Nothing is silently overwritten.
CREATE TABLE IF NOT EXISTS edge_audit (
    id            BIGSERIAL PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'default',
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    old_weight    REAL,
    new_weight    REAL NOT NULL,
    reason        TEXT NOT NULL,
    components    JSONB,
    at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS edge_audit_edge ON edge_audit (source_id, target_id);

CREATE TABLE IF NOT EXISTS watch_subscriptions (
    id            BIGSERIAL PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'default',
    categories    TEXT[] NOT NULL,
    similarity_floor REAL NOT NULL DEFAULT 0.45,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS watch_queue (
    id            BIGSERIAL PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'default',
    arxiv_id      TEXT NOT NULL,
    title         TEXT NOT NULL,
    similarity    REAL NOT NULL,
    nearest_paper_id TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, arxiv_id)
);

CREATE TABLE IF NOT EXISTS digests (
    id            BIGSERIAL PRIMARY KEY,
    workspace_id  TEXT NOT NULL DEFAULT 'default',
    period_label  TEXT NOT NULL,
    report        JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
