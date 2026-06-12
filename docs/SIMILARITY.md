# The similarity function

The weighted `RELATED_TO` edge is Lattice's signature output and its core IP.
This document is the precise specification of how edge weights are computed,
calibrated, and audited. Implementation: `backend/lattice/graph/similarity.py`.

## The formula

For two papers `i` and `j`:

```
w(i, j) = sigma( a*S_sem + b*S_meth + c*S_cit + d*S_data ) * T(i, j)
```

Each component `S_*` is normalized to `[0, 1]`. The linear combination is squashed
by a logistic sigmoid, then modulated by a temporal factor `T`. Defaults:
`a=0.40, b=0.25, c=0.25, d=0.10` (all in `config.py`, tunable, eval-measured).

Edges below `tau = 0.35` are not materialized (sparsity control), and each paper
keeps at most `knn_cap = 15` outgoing `RELATED_TO` edges (legibility).

## Components

### S_sem - thematic similarity
Cosine similarity of SPECTER2 (title + abstract) embeddings, **corpus-calibrated**.

SPECTER-family models discriminate *related* pairs better than they reject
*unrelated* ones, so raw cosine is optimistically high and bunched. The
`CosineCalibrator` rescales raw cosine against the corpus distribution using
robust percentiles (p5 to p95) so the score spreads across `[0, 1]` and `tau`
means the same thing across corpora. Until enough pairs exist to fit, it is an
identity clamp (a safe default).

### S_meth - methodological proximity
A blend of two signals:

- Jaccard over normalized `methods_taxonomy` tags (lexical, precise).
- Cosine over the **methodology-section** aspect embeddings (semantic).

`S_meth = w_sec * cos(method_emb) + (1 - w_sec) * jaccard(tags)` with
`w_sec = 0.6`. Section-level comparison materially improves discrimination over
title+abstract alone. If only one signal is available it is used directly.

### S_cit - citation structure
`max(bibliographic_coupling, co_citation)`, saturated to `1.0` when a direct
citation exists between the two papers.

- Bibliographic coupling: Jaccard over the two papers' reference id sets.
- Co-citation: how often the two are cited together elsewhere (Semantic Scholar).

### S_data - dataset overlap
Jaccard over resolved `Dataset` nodes. Small weight, but a high-precision signal:
two papers sharing a dataset are very likely comparable.

## Graceful degradation

Citation and dataset signals often arrive late or not at all (enrichment is
asynchronous and rate-limited). When a component's inputs are missing, it is
**dropped** and the remaining coefficients are **renormalized** so the score is
not unfairly deflated. The set of components that actually contributed is recorded
in `weight_components.available`, so every edge is honest about what it was
computed from. With only SPECTER available, the system degrades to text-only
similarity rather than producing a misleadingly low weight.

## The temporal factor T

Old foundational links are **not** decayed - they still matter. `T` only applies
a recency *boost* (`1 + 0.15`) to edges touching a paper ingested within the last
`90` days, so "what's new" views surface fresh connections without distorting the
historical structure.

## Auditability

Every computation returns a `WeightResult` with the final weight, the pre-sigmoid
raw score, the temporal factor, each component value, the components that were
available, and the effective (renormalized) coefficients. This is stored on the
edge as `weight_components` (JSON) and shown in the UI's "edge anatomy" panel.
Every change writes to `edge_audit` with old and new weights and a reason; nothing
is silently overwritten, and superseded edges get `invalid_at` set rather than
being deleted (bi-temporal).

## Incremental update (the evolving graph)

On ingesting paper `p` (`graph/evolution.py`, wired in `ingestion/service.py`):

1. Candidate generation: top `candidate_k = 50` by SPECTER ANN (pgvector). Never
   all-pairs.
2. Compute the full composite only against candidates.
3. Materialize edges above `tau`, sorted by weight, capped at `knn_cap`.
4. Re-resolve entities: a new method/dataset that fuzzy-matches an existing node
   is MERGEd (`graph/entity_resolution.py`).
5. Citation enrichment can arrive later; a second pass updates the affected
   `S_cit` components and re-audits.

This makes ingestion `O(k)` per paper rather than `O(n^2)`, append-mostly, and
fully auditable.

## Tuning

`eval/edge_quality.py` reports the correlation between `w(i,j)` and human/LLM
judged relatedness, plus precision@k for "most related" lists, **per weight
configuration**. Tuning `(a, b, c, d)` is therefore a measured decision, not a
vibe: change the coefficients in config, re-run the edge-quality eval, keep the
configuration that wins.
