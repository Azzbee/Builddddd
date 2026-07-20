"""Ingestion service: wires every stage into the resumable pipeline.

This is where M1-M4 come together. Each stage handler is a method that reads and
populates the shared PipelineContext. All collaborators are injected (parser, LLM,
enrichment, embedders, stores, graph writer), so the entire pipeline runs and is
tested offline with in-memory stores and a scripted model. Linking is the final
stage and touches the graph last, so a crash before it leaves no paper in the
graph at all; its writes are idempotent MERGEs (nodes, edges, and the audit trail),
so a crash partway through linking self-heals on re-run rather than duplicating.
Note: linking spans two stores (graph + card/blob) and is not a single ACID
transaction, so the self-healing relies on that idempotency, not on rollback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from lattice.config import Settings
from lattice.core.cost import CostTracker
from lattice.core.errors import DuplicateError
from lattice.core.hashing import content_hash, normalize_arxiv, normalize_doi, stable_id
from lattice.core.llm import LLMClient
from lattice.core.logging import get_logger
from lattice.db.blobs import BlobStore, InMemoryBlobStore
from lattice.db.cards import CorpusStore, InMemoryJobStore, JobStore, StoredFeatures
from lattice.db.ingest_artifacts import (
    IngestArtifactStore,
    InMemoryIngestArtifactStore,
)
from lattice.db.vector import VectorRecord, VectorStore
from lattice.embeddings.chunks import AspectEmbedder, ChunkEmbedder
from lattice.embeddings.specter2 import Specter2Embedder
from lattice.extraction.extractor import extract_paper_card
from lattice.extraction.schemas import Author, PaperCard
from lattice.graph.contradictions import (
    ClaimEdge,
    ClaimRecord,
    HeuristicNLIJudge,
    NLIJudge,
    detect_relations,
    detect_relations_for,
)
from lattice.graph.entity_resolution import EntityResolver
from lattice.graph.evolution import (
    CitationSignal,
    EdgeUpdate,
    compute_related_edges,
    detect_supersession,
    diff_audit,
)
from lattice.graph.models import GraphEdge, GraphNode, GraphSnapshot
from lattice.graph.reader import GraphReader
from lattice.graph.similarity import CosineCalibrator, PaperFeatures, cosine
from lattice.graph.store import FakeGraphStore, GraphStore
from lattice.graph.writer import GraphWriter
from lattice.ingestion.chunker import chunk_document
from lattice.ingestion.dedup import PaperIdentity
from lattice.ingestion.models import (
    IngestJob,
    JobStage,
    JobStatus,
    ParsedDocument,
    SourceType,
)
from lattice.ingestion.pdf_utils import classify_pdf
from lattice.ingestion.pipeline import IngestionPipeline, PipelineContext
from lattice.landscape.proposal import FacetPaper, MomentumLite

log = get_logger("ingestion.service")


class Parser(Protocol):
    async def process_fulltext(self, pdf_bytes: bytes, filename: str = ...) -> ParsedDocument: ...


class Enricher(Protocol):
    async def enrich(self, card: PaperCard) -> dict[str, object]: ...


def assign_paper_id(doc: ParsedDocument, workspace_id: str) -> str:
    """Deterministic paper id from the strongest available identifier."""
    doi = normalize_doi(doc.doi)
    if doi:
        return stable_id(workspace_id, "doi", doi)
    arxiv = normalize_arxiv(doc.arxiv_id)
    if arxiv:
        return stable_id(workspace_id, "arxiv", arxiv)
    first_author = doc.authors[0] if doc.authors else ""
    return stable_id(workspace_id, "title", doc.title, first_author)


@dataclass
class IngestionService:
    settings: Settings
    llm: LLMClient
    parser: Parser
    vectors: VectorStore
    cards: CorpusStore
    jobs: JobStore = field(default_factory=InMemoryJobStore)
    artifacts: IngestArtifactStore = field(default_factory=InMemoryIngestArtifactStore)
    #: Raw PDF storage for the in-app reader + citation deep-linking.
    blobs: BlobStore = field(default_factory=InMemoryBlobStore)
    graph: GraphStore = field(default_factory=FakeGraphStore)
    #: Optional Neo4j-backed reader; when set, read paths query the live graph
    #: instead of the in-memory mirror (production cross-process correctness).
    reader: GraphReader | None = None
    enricher: Enricher | None = None
    specter: Specter2Embedder = field(default_factory=lambda: Specter2Embedder(dim=768))
    chunk_embedder: ChunkEmbedder = field(default_factory=lambda: ChunkEmbedder(dim=1024))
    aspect_embedder: AspectEmbedder = field(default_factory=lambda: AspectEmbedder(dim=1024))

    # Per-paper feature state for O(k) incremental linking. Rehydrated from the
    # persistent stores once per process by hydrate() (see below), so a restart does
    # not leave a new paper linking against an empty pool.
    _features: dict[str, PaperFeatures] = field(default_factory=dict)
    _external_ids: dict[str, set[str]] = field(default_factory=dict)
    #: In-memory edge mirror so the explorer works without Neo4j (demo/dev mode).
    _related: dict[str, list[EdgeUpdate]] = field(default_factory=dict)
    #: In-memory claim-relation mirror (contradictions/convergence) for the API.
    _claim_relations: list[ClaimEdge] = field(default_factory=list)
    #: Per-paper aspect embeddings (problem/methodology/results) for quadrants.
    _aspects: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    #: In-memory directed citation edges (src cites dst) for lineage.
    _cites: list[tuple[str, str]] = field(default_factory=list)
    method_resolver: EntityResolver | None = None
    dataset_resolver: EntityResolver | None = None
    #: Text extractor for PDF sanity checks (defaults to pypdf via classify_pdf).
    text_extractor: Callable[[bytes], str] | None = None
    #: Whether the in-memory linking state has been rehydrated from persistent stores
    #: this process. Guards a one-shot load so restarts don't link against an empty pool.
    _hydrated: bool = False
    _hydrate_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        ws = self.settings.workspace_id
        self.method_resolver = self.method_resolver or EntityResolver("method", ws)
        self.dataset_resolver = self.dataset_resolver or EntityResolver("dataset", ws)
        self._writer = GraphWriter(self.graph, ws)

    # ------------------------------------------------------------------ public
    async def hydrate(self) -> None:
        """Rebuild the in-memory linking state from the persistent stores.

        Incremental linking keeps a per-paper feature pool (SPECTER + aspect vectors,
        method and dataset sets) and two entity-resolver registries in memory. Those
        are lost on a process restart, which would leave a freshly ingested paper
        linking against an empty pool and minting duplicate Method/Dataset nodes.
        This loads them back from the card store exactly once per process, before the
        first ingest. It is idempotent: papers already in the in-memory pool are
        kept, and superseded papers are excluded (load_features filters them).

        Rehydrated papers link at full similarity fidelity: SPECTER (sem),
        methodology-section vectors (meth), method tags, datasets, citation
        reference sets (S_cit bibliographic coupling), and external ids (direct-
        citation detection) all persist and restore. Nothing degrades on restart.
        """
        if self._hydrated:
            return
        # Serialize hydration: a second concurrent ingest must WAIT for the in-flight
        # load rather than race past a half-populated pool. Double-check inside the
        # lock so only the first waiter does the work.
        async with self._hydrate_lock:
            if self._hydrated:
                return
            try:
                stored = await self.cards.load_features()
            except Exception as exc:  # persistence hiccup: run with what's in memory
                log.warning("hydrate.failed", error=str(exc))
                self._hydrated = True  # don't retry-loop on every ingest
                return
            self._populate_from_features(stored)
            self._hydrated = True
            log.info("hydrate.done", papers=len(stored))

    def _populate_from_features(self, stored: list[StoredFeatures]) -> None:
        import numpy as np

        for feat in stored:
            if feat.paper_id in self._features:
                continue  # keep the richer in-memory features from this process
            meth_vec = (feat.aspects or {}).get("methodology")
            self._features[feat.paper_id] = PaperFeatures(
                paper_id=feat.paper_id,
                specter=(np.asarray(feat.specter, dtype=float) if feat.specter else None),
                methodology_embedding=(np.asarray(meth_vec, dtype=float) if meth_vec else None),
                methods=feat.methods,
                datasets=feat.datasets,
                references=set(feat.references),
            )
            if feat.external_ids:
                # Direct-citation detection needs the candidate's own ids too.
                self._external_ids.setdefault(feat.paper_id, set(feat.external_ids))
            if feat.aspects:
                # Quadrants (cross-community transfer) survive restarts too.
                self._aspects.setdefault(feat.paper_id, feat.aspects)
        # Re-register canonical entities so fuzzy resolution is stable across restarts:
        # a variant name now resolves to the same node it did before the restart.
        assert self.method_resolver is not None and self.dataset_resolver is not None
        for feat in stored:
            for m in feat.methods:
                self.method_resolver.register(m)
            for d in feat.datasets:
                self.dataset_resolver.register(d)

    async def ingest_pdf(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        await self.hydrate()
        job = await self.stage_pdf(source_ref, pdf_bytes)
        if job.status in (JobStatus.SUCCEEDED, JobStatus.DUPLICATE):
            return job
        saved = await self.artifacts.get(job.job_id)
        ctx = PipelineContext(job=job, raw_pdf=pdf_bytes)
        if saved is not None:
            saved.restore(ctx)
            ctx.raw_pdf = ctx.raw_pdf or pdf_bytes
        return await self._run_context(ctx)

    async def stage_pdf(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        """Persist a deterministic queued job and its source without processing it."""
        digest = content_hash(pdf_bytes)
        job_id = stable_id("job", self.settings.workspace_id, source_ref, digest)
        existing = await self.jobs.get(job_id)
        if existing is not None:
            return existing
        job = IngestJob(
            job_id=job_id,
            workspace_id=self.settings.workspace_id,
            source_type=SourceType.FILE,
            source_ref=source_ref,
            content_hash=digest,
        )
        ctx = PipelineContext(job=job, raw_pdf=pdf_bytes)
        await self.jobs.save(job)
        await self.artifacts.save_context(ctx)
        return job

    async def resume_job(self, job_id: str) -> IngestJob | None:
        """Resume a persisted paused or failed job from its last completed stage."""
        await self.hydrate()
        job = await self.jobs.get(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.SUCCEEDED, JobStatus.DUPLICATE):
            return job
        saved = await self.artifacts.get(job_id)
        if saved is None or saved.raw_pdf is None:
            return job
        ctx = PipelineContext(job=job)
        saved.restore(ctx)
        return await self._run_context(ctx)

    async def _run_context(self, ctx: PipelineContext) -> IngestJob:
        await self._restore_runtime_state(ctx)
        ctx.extra["cost"] = CostTracker(
            per_job_cap=self.settings.cost.per_job_usd_cap,
            daily_cap=self.settings.cost.daily_usd_cap,
        )
        cost = ctx.extra["cost"]
        if isinstance(cost, CostTracker):
            cost.job_usage.cost_usd = ctx.job.cost_usd
        pipeline = IngestionPipeline(
            {
                JobStage.PARSING: self._parse,
                JobStage.EXTRACTING: self._extract,
                JobStage.ENRICHING: self._enrich,
                JobStage.EMBEDDING: self._embed,
                JobStage.LINKING: self._link,
            },
            store=self.jobs,
            artifact_store=self.artifacts,
            max_attempts=self.settings.ingest_max_attempts,
        )
        return await pipeline.run(ctx)

    async def _restore_runtime_state(self, ctx: PipelineContext) -> None:
        """Restore process-local embedding state needed by the linking stage."""
        feature = ctx.extra.get("_paper_feature")
        if not isinstance(feature, dict) or ctx.job.paper_id is None:
            return
        import numpy as np

        pid = ctx.job.paper_id
        specter = feature.get("specter")
        methodology = feature.get("methodology_embedding")
        self._features[pid] = PaperFeatures(
            paper_id=pid,
            specter=np.asarray(specter, dtype=float) if isinstance(specter, list) else None,
            methodology_embedding=(
                np.asarray(methodology, dtype=float) if isinstance(methodology, list) else None
            ),
            methods=set(feature.get("methods", [])),
            datasets=set(feature.get("datasets", [])),
            references=set(feature.get("references", [])),
            ingested_at=datetime.fromisoformat(str(feature["ingested_at"])),
        )
        external_ids = ctx.extra.get("_external_ids")
        if isinstance(external_ids, list):
            self._external_ids[pid] = {str(value) for value in external_ids}
        aspects = ctx.extra.get("_aspects")
        if isinstance(aspects, dict):
            self._aspects[pid] = {
                str(name): [float(value) for value in values]
                for name, values in aspects.items()
                if isinstance(values, list)
            }
        record_values = ctx.extra.get("_chunk_records")
        if isinstance(record_values, list):
            records = [VectorRecord(**value) for value in record_values if isinstance(value, dict)]
            await self.vectors.upsert_chunks(records)

    # ------------------------------------------------------------------ stages
    async def _parse(self, ctx: PipelineContext) -> None:
        assert ctx.raw_pdf is not None
        text = classify_pdf(ctx.raw_pdf, self.text_extractor)  # raises typed errors
        doc = await self.parser.process_fulltext(ctx.raw_pdf, ctx.job.source_ref)
        if not doc.full_text().strip():
            # Parser produced nothing usable; fall back to raw text as one section.
            doc.abstract = doc.abstract or text[:2000]
        ctx.document = doc
        paper_id = assign_paper_id(doc, ctx.job.workspace_id)
        ctx.job.paper_id = paper_id

        # Dedup against the corpus (idempotent no-op on re-ingest).
        index = await self.cards.corpus_index()
        index.add(
            PaperIdentity(
                paper_id, doc.title, doc.authors, doc.doi, doc.arxiv_id, ctx.job.content_hash
            )
        )
        candidate = PaperIdentity(
            paper_id="__incoming__",
            title=doc.title,
            authors=doc.authors,
            doi=doc.doi,
            arxiv_id=doc.arxiv_id,
            content_hash=ctx.job.content_hash,
        )
        # Rebuild index without the just-added incoming entry for the check.
        check_index = await self.cards.corpus_index()
        dup = check_index.find_duplicate(candidate)
        if dup.is_duplicate:
            # A title-level match can be the same manuscript in a different VERSION
            # (arXiv preprint vs the published, DOI-bearing article). That is a
            # supersession, not a duplicate: the newer version should be ingested
            # and the older one superseded, never silently rejected. Identifier
            # matches (content hash / DOI / arXiv id) are the same artifact and
            # stay duplicates.
            supersession = None
            if dup.reason in ("title_author_fuzzy", "title_exact") and dup.existing_paper_id:
                existing = await self.cards.get(dup.existing_paper_id)
                if existing is not None:
                    new_has_doi = bool(normalize_doi(doc.doi))
                    new_is_preprint = bool(normalize_arxiv(doc.arxiv_id)) and not new_has_doi
                    supersession = detect_supersession(
                        paper_id,
                        new_has_doi,
                        new_is_preprint,
                        (
                            existing.paper_id,
                            bool(existing.doi),
                            bool(existing.arxiv_id) and not existing.doi,
                        ),
                    )
            if supersession is not None:
                superseded_id, superseding_id = supersession
                if superseding_id == paper_id:
                    # The incoming paper is the published version: proceed with the
                    # ingest; _link applies the supersession after the paper lands.
                    ctx.extra["supersedes"] = superseded_id
                    log.info(
                        "supersession.detected", superseded=superseded_id, superseding=paper_id
                    )
                    return
                ctx.extra["duplicate_of"] = dup.existing_paper_id
                raise DuplicateError(
                    f"outdated preprint of {dup.existing_paper_id} "
                    "(published version already in corpus)"
                )
            ctx.extra["duplicate_of"] = dup.existing_paper_id
            raise DuplicateError(f"duplicate of {dup.existing_paper_id} ({dup.reason})")

    async def _extract(self, ctx: PipelineContext) -> None:
        assert ctx.document is not None and ctx.job.paper_id is not None
        doc = ctx.document
        cost = ctx.extra.get("cost")
        identity: dict[str, object] = {
            "paper_id": ctx.job.paper_id,
            "title": doc.title,
            "authors": [Author(name=a) for a in doc.authors],
            "year": doc.year,
            "venue": doc.venue,
            "doi": normalize_doi(doc.doi),
            "arxiv_id": normalize_arxiv(doc.arxiv_id),
            "abstract": doc.abstract,
        }
        card = await extract_paper_card(
            identity=identity,
            body=doc.full_text(),
            parse_confidence=doc.overall_confidence,
            llm=self.llm,
            settings=self.settings.extraction,
            cost=cost if isinstance(cost, CostTracker) else None,
        )
        ctx.card = card

    async def _enrich(self, ctx: PipelineContext) -> None:
        # Best-effort: enrichment failures degrade to text-only similarity.
        assert ctx.card is not None
        if self.enricher is None:
            return
        try:
            extra = await self.enricher.enrich(ctx.card)
        except Exception as exc:
            log.warning("enrich.failed", paper_id=ctx.card.paper_id, error=str(exc))
            return
        ctx.extra.update(extra)
        if extra.get("s2_paper_id"):
            ctx.card.s2_paper_id = str(extra["s2_paper_id"])
        if extra.get("openalex_id"):
            # Keeps the paper matchable against OpenAlex-sourced reference lists
            # (they cite by https://openalex.org/W... URL, not DOI).
            ctx.card.openalex_id = str(extra["openalex_id"])

    async def _embed(self, ctx: PipelineContext) -> None:
        assert ctx.card is not None and ctx.document is not None
        card = ctx.card
        pid = card.paper_id

        precomputed = ctx.extra.get("specter_embedding")
        paper_emb = self.specter.embed(
            card.title,
            card.abstract,
            precomputed=precomputed,  # type: ignore[arg-type]
        )
        aspects = self.aspect_embedder.embed_card(card)

        chunks = chunk_document(ctx.document, pid)
        chunk_vecs = self.chunk_embedder.embed_chunks(chunks)
        records = [
            VectorRecord(
                chunk_id=c.chunk_id,
                paper_id=pid,
                workspace_id=card.__dict__.get("workspace_id", self.settings.workspace_id),
                title=card.title,
                section_title=c.section_title,
                text=c.text,
                embedding=chunk_vecs[c.chunk_id],
                evidence_location=c.section_title,
                page=c.page,
            )
            for c in chunks
        ]
        await self.vectors.upsert_chunks(records)
        ctx.chunks = chunks

        import numpy as np

        refs_value = ctx.extra.get("reference_ids")
        refs = set(refs_value) if isinstance(refs_value, list | set) else set()
        ingested_at = datetime.now(UTC)
        self._features[pid] = PaperFeatures(
            paper_id=pid,
            specter=np.asarray(paper_emb.vector, dtype=float),
            methodology_embedding=np.asarray(aspects["methodology"], dtype=float),
            methods=card.normalized_methods,
            datasets=card.normalized_datasets,
            references=refs,
            ingested_at=ingested_at,
        )
        self._external_ids[pid] = card.external_ids
        self._aspects[pid] = aspects
        ctx.extra["_paper_feature"] = {
            "specter": paper_emb.vector,
            "methodology_embedding": aspects["methodology"],
            "methods": sorted(card.normalized_methods),
            "datasets": sorted(card.normalized_datasets),
            "references": sorted(refs),
            "ingested_at": ingested_at.isoformat(),
        }
        ctx.extra["_external_ids"] = sorted(card.external_ids)
        ctx.extra["_aspects"] = aspects
        ctx.extra["_chunk_records"] = [record.__dict__ for record in records]

    async def _link(self, ctx: PipelineContext) -> None:
        assert ctx.card is not None and ctx.job.paper_id is not None
        card = ctx.card
        pid = card.paper_id
        new_feat = self._features[pid]

        # Candidate generation: top-k by SPECTER cosine among existing papers. A
        # paper this ingest supersedes is excluded - the published version must not
        # link to its own preprint.
        superseded_id = ctx.extra.get("supersedes")
        candidates = [c for c in self._ann_candidates(pid, new_feat) if c.paper_id != superseded_id]
        calibrator = self._fit_calibrator()
        citations = self._citation_signals(new_feat, candidates)

        edges = compute_related_edges(
            new_feat,
            candidates,
            self.settings.similarity,
            calibrator,
            citations=citations,
        )

        # Persist: paper node, entities, claims, then edges + audit (last, atomic).
        await self._writer.upsert_paper(card)
        for author in card.authors:
            await self._writer.upsert_author(author.name, pid, s2_id=author.s2_id)
        method_names = card.methods_taxonomy + card.methodology.techniques
        method_embs = self.chunk_embedder.embed_texts(method_names) if method_names else []
        for method, emb in zip(method_names, method_embs, strict=True):
            res = self.method_resolver.resolve(method, embedding=emb)  # type: ignore[union-attr]
            await self._writer.upsert_method(res.key, res.name, pid)
        dataset_names = [ds.name for ds in card.datasets]
        dataset_embs = self.chunk_embedder.embed_texts(dataset_names) if dataset_names else []
        for ds_name, emb in zip(dataset_names, dataset_embs, strict=True):
            res = self.dataset_resolver.resolve(ds_name, embedding=emb)  # type: ignore[union-attr]
            await self._writer.upsert_dataset(res.key, ds_name, pid)
        for concept in card.domains:
            await self._writer.upsert_concept(
                stable_id(self.settings.workspace_id, "concept", concept), concept, pid
            )
        for result in card.key_results:
            await self._writer.upsert_claim(pid, result.claim, result.evidence_location)

        existing = await self._writer.existing_related_weights(pid)
        audit = diff_audit(edges, existing)
        for edge in edges:
            await self._writer.upsert_related_edge(edge)
        for sig_pid, sig in citations.items():
            if sig.direct_citation:
                await self._writer.upsert_cites(pid, sig_pid)
                self._cites.append((pid, sig_pid))
        if audit:
            await self._writer.write_audit(
                [
                    {
                        "src": a.source_id,
                        "dst": a.target_id,
                        "old": a.old_weight,
                        "new": a.new_weight,
                        "reason": a.reason,
                        "at": a.at.isoformat(),
                    }
                    for a in audit
                ]
            )

        # Persist the SPECTER vector and aspect embeddings alongside the card so
        # incremental linking rehydrates at full similarity fidelity after a restart
        # (and the epistemic quadrants keep their aspect vectors).
        specter = new_feat.specter.tolist() if new_feat.specter is not None else None
        await self.cards.put_card(
            card,
            specter=specter,
            aspects=self._aspects.get(pid),
            references=sorted(new_feat.references),
            content_hash=ctx.job.content_hash,
        )
        # Persist the source PDF last (after the card exists, for the FK), so the
        # reader can open it and citations can deep-link to a page.
        if ctx.raw_pdf:
            meta = await self.blobs.put(pid, ctx.raw_pdf, content_hash=ctx.job.content_hash)
            ctx.extra["pdf_pages"] = meta.pages
        self._related[pid] = edges
        ctx.extra["edges_written"] = len(edges)

        if isinstance(superseded_id, str):
            await self._apply_supersession(superseded_id, pid)

        # Living graph: contradictions/support surface as papers arrive, not only
        # on a manual full-corpus pass. (After supersession, so a replaced preprint's
        # claims are already out of the comparison set.)
        if self.settings.incremental_contradictions and card.key_results:
            await self._detect_relations_incremental(card)

    async def _apply_supersession(self, old_id: str, new_id: str) -> None:
        """Supersede ``old_id`` with ``new_id`` (preprint -> published version).

        Bi-temporal: the old paper and its history stay in every store; its edges
        get ``invalid_at`` set (both directions) so default views hide them, a
        SUPERSEDED_BY edge records the succession, and the old paper leaves the
        linking candidate pool and the analytics so the same work is never counted
        twice. All writes are idempotent (MERGE / UPDATE / set-discard).
        """
        await self._writer.set_superseded(old_id, new_id)
        await self._writer.invalidate_related_edges_of(old_id, reason=f"superseded_by:{new_id}")
        await self.cards.mark_superseded(old_id, new_id)
        # Leave the in-memory linking pool and mirrors.
        self._features.pop(old_id, None)
        self._external_ids.pop(old_id, None)
        self._aspects.pop(old_id, None)
        self._related.pop(old_id, None)
        for edge_list in self._related.values():
            edge_list[:] = [e for e in edge_list if e.target_id != old_id]
        self._claim_relations = [
            e for e in self._claim_relations if old_id not in (e.source_paper, e.target_paper)
        ]
        log.info("supersession.applied", superseded=old_id, superseding=new_id)

    async def _active_cards(self) -> list[PaperCard]:
        """All cards minus superseded versions.

        Analytics, exports, and default views must never count the same work twice
        (a preprint and its published successor). Direct by-id reads intentionally
        bypass this - superseded papers remain retrievable, just not aggregated.
        """
        cards = await self.cards.all_cards()
        superseded = await self.cards.superseded_map()
        if not superseded:
            return cards
        return [c for c in cards if c.paper_id not in superseded]

    # ------------------------------------------------------------------ contradictions
    def _claim_records(self, cards: list[PaperCard]) -> list[ClaimRecord]:
        """Flatten cards' key results into concept-keyed ClaimRecords for the judge."""
        ws = self.settings.workspace_id
        claims: list[ClaimRecord] = []
        for card in cards:
            authors = frozenset(a.normalized_name for a in card.authors)
            concepts = card.domains or ["general"]
            for result in card.key_results:
                for concept in concepts:
                    claims.append(
                        ClaimRecord(
                            claim_id=stable_id(ws, card.paper_id, result.claim),
                            paper_id=card.paper_id,
                            text=result.claim,
                            concept=concept,
                            authors=authors,
                            year=card.year,
                            evidence_location=result.evidence_location,
                        )
                    )
        return claims

    async def _detect_relations_incremental(self, card: PaperCard) -> None:
        """Judge the new paper's claims against existing same-concept claims.

        Runs at the end of every ingest (offline heuristic judge: free,
        deterministic), so SUPPORTS/CONTRADICTS/EXTENDS edges accumulate as papers
        arrive - the living graph - instead of waiting for a manual full-corpus
        pass. Idempotent: graph writes are MERGEs and the in-memory mirror dedupes
        by (source, target, relation).
        """
        existing_cards = [c for c in await self._active_cards() if c.paper_id != card.paper_id]
        if not existing_cards:
            return
        report = await detect_relations_for(
            self._claim_records([card]),
            self._claim_records(existing_cards),
            HeuristicNLIJudge(),
        )
        seen = {(e.source_id, e.target_id, str(e.relation)) for e in self._claim_relations}
        added = 0
        for edge in report.edges:
            key = (edge.source_id, edge.target_id, str(edge.relation))
            if key in seen:
                continue
            seen.add(key)
            await self._writer.upsert_claim_relation(
                edge.source_id, edge.target_id, str(edge.relation), edge.confidence
            )
            self._claim_relations.append(edge)
            added += 1
        if added:
            log.info("contradictions.incremental", paper_id=card.paper_id, edges=added)

    async def analyze_relations(self, judge: NLIJudge | None = None) -> list[ClaimEdge]:
        """Detect SUPPORTS/CONTRADICTS/EXTENDS relations across the corpus' claims.

        Groups every card's key results into ClaimRecords keyed by concept, runs the
        judge over same-concept cross-paper pairs, persists the resulting edges to
        the graph, and mirrors them in memory for the API. Defaults to the offline
        heuristic judge; pass an LLM judge in production.
        """
        judge = judge or HeuristicNLIJudge()
        claims = self._claim_records(await self._active_cards())
        report = await detect_relations(claims, judge)
        # Persist to the graph and mirror in memory (dedupe by id pair + relation).
        seen: set[tuple[str, str, str]] = set()
        for edge in report.edges:
            key = (edge.source_id, edge.target_id, str(edge.relation))
            if key in seen:
                continue
            seen.add(key)
            await self._writer.upsert_claim_relation(
                edge.source_id, edge.target_id, str(edge.relation), edge.confidence
            )
        self._claim_relations = report.edges
        log.info(
            "contradictions.analyzed",
            claims=len(claims),
            contradictions=len(report.contradictions),
            supports=len(report.supports),
        )
        return report.edges

    async def get_claim_relations(self, relation: str | None = None) -> list[ClaimEdge]:
        """Claim relations from the live graph (reader) or the in-memory mirror."""
        if self.reader is not None:
            return await self.reader.claim_relations(self.settings.workspace_id, relation)
        edges = self._claim_relations
        if relation:
            edges = [e for e in edges if str(e.relation) == relation.upper()]
        return edges

    # ------------------------------------------------------------------ quadrants
    def _adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {}
        for src, edges in self._related.items():
            for e in edges:
                adj.setdefault(src, set()).add(e.target_id)
                adj.setdefault(e.target_id, set()).add(src)
        return adj

    def _distance(self, adj: dict[str, set[str]], a: str, b: str, cap: int = 6) -> int:
        if a == b:
            return 0
        seen = {a}
        frontier = [a]
        for d in range(1, cap + 1):
            nxt: list[str] = []
            for node in frontier:
                for nb in adj.get(node, set()):
                    if nb == b:
                        return d
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            if not nxt:
                break
            frontier = nxt
        return cap + 1  # effectively disconnected / far

    async def epistemic_quadrants(self) -> dict[str, object]:
        """Compute the four epistemic quadrants from the corpus.

        Known knowns (independently supported, non-contradicted claims), known
        unknowns (clustered open problems), and unknown knowns (cross-community
        transfer candidates). Unknown unknowns are surfaced as proxies elsewhere
        (high-pressure empty matrix cells).
        """
        from lattice.landscape.quadrants import (
            OpenProblemMention,
            SupportingPaper,
            cluster_claims,
            cluster_open_problems,
            consolidate_known_knowns,
            cross_community_transfer,
        )

        cards = await self._active_cards()
        now_year = datetime.now(UTC).year

        # Known knowns: fuzzily cluster claims that state the same finding (exact-text
        # grouping reports zero on real corpora), then keep the independently
        # supported, non-contradicted ones.
        from lattice.core.hashing import normalize_text

        items: list[tuple[str, SupportingPaper]] = []
        for card in cards:
            authors = frozenset(a.normalized_name for a in card.authors)
            method = card.methods_taxonomy[0] if card.methods_taxonomy else None
            dataset = card.datasets[0].name if card.datasets else None
            for r in card.key_results:
                if r.claim.strip():
                    items.append(
                        (
                            r.claim,
                            SupportingPaper(card.paper_id, authors, method, dataset, card.year),
                        )
                    )
        claim_clusters = cluster_claims(items)
        contradictions = await self.get_claim_relations("CONTRADICTS")
        contra_norm = {normalize_text(e.source_text) for e in contradictions} | {
            normalize_text(e.target_text) for e in contradictions
        }
        # Drop the contradicted *members* of each cluster (a single contested claim
        # should not nuke an otherwise-converged finding), then keep clusters that
        # still have independent support.
        grouped: dict[str, list[SupportingPaper]] = {}
        for cl in claim_clusters:
            survivors = [
                (m, p)
                for m, p in zip(cl.members, cl.papers, strict=True)
                if normalize_text(m) not in contra_norm
            ]
            if not survivors:
                continue
            canonical = max((m for m, _ in survivors), key=len)
            grouped[canonical] = [p for _, p in survivors]
        findings = consolidate_known_knowns(grouped)
        known_knowns = [
            {
                "claim": f.claim,
                "independent_supports": f.independent_supports,
                "triangulated": f.triangulated,
                "latest_year": f.latest_year,
                "strength": round(f.strength, 3),
            }
            for f in findings
        ]

        # Known unknowns: cluster open problems from future_work.
        mentions: list[OpenProblemMention] = []
        fw_texts = [(c.paper_id, fw, c.year) for c in cards for fw in c.future_work if fw.strip()]
        if fw_texts:
            vecs = self.chunk_embedder.embed_texts([t for _p, t, _y in fw_texts])
            for (pid, text, year), vec in zip(fw_texts, vecs, strict=True):
                mentions.append(OpenProblemMention(pid, text, vec, year))
        clusters = cluster_open_problems(mentions, current_year=now_year)
        known_unknowns = [
            {
                "problem": cl.canonical_text,
                "frequency": cl.frequency,
                "latest_year": cl.latest_year,
                "span_years": cl.span_years,
                "score": round(cl.score, 3),
            }
            for cl in clusters[:20]
        ]

        # Unknown knowns: cross-community transfer (method next door to a problem).
        adj = self._adjacency()
        sources = [
            (
                card.methods_taxonomy[0] if card.methods_taxonomy else "method",
                card.paper_id,
                self._aspects[card.paper_id]["methodology"],
            )
            for card in cards
            if card.paper_id in self._aspects
        ]
        targets = [
            (card.paper_id, self._aspects[card.paper_id]["problem"])
            for card in cards
            if card.paper_id in self._aspects
        ]
        transfers = cross_community_transfer(
            sources, targets, distance_fn=lambda a, b: self._distance(adj, a, b)
        )
        unknown_knowns = [
            {
                "method": t.method,
                "source_paper": t.source_paper,
                "target_paper": t.target_paper,
                "similarity": round(t.similarity, 3),
                "graph_distance": t.graph_distance,
            }
            for t in transfers[:20]
        ]

        return {
            "known_knowns": known_knowns,
            "known_unknowns": known_unknowns,
            "unknown_knowns": unknown_knowns,
        }

    # ------------------------------------------------------------------ proposals
    def _facet_papers(
        self, cards: list[PaperCard], row_facet: str, col_facet: str
    ) -> list[FacetPaper]:
        """Project cards onto two facets for the proposal generator."""

        def facet(card: PaperCard, name: str) -> set[str]:
            if name == "method":
                return card.normalized_methods
            if name == "dataset":
                return card.normalized_datasets
            if name == "concept":
                return {d.lower() for d in card.domains}
            raise ValueError(f"unknown facet: {name}")

        return [
            FacetPaper(
                paper_id=card.paper_id,
                title=card.title,
                year=card.year,
                row_values=facet(card, row_facet),
                col_values=facet(card, col_facet),
                representative_claim=card.key_results[0].claim if card.key_results else None,
                limitations=list(card.limitations),
            )
            for card in cards
        ]

    def _facet_momentum(self, cards: list[PaperCard], facet: str, value: str) -> MomentumLite:
        """MomentumLite for papers whose ``facet`` includes ``value``."""
        from lattice.landscape.momentum import momentum_score

        def has(card: PaperCard) -> bool:
            if facet == "method":
                return value in card.normalized_methods
            if facet == "dataset":
                return value in card.normalized_datasets
            return value in {d.lower() for d in card.domains}

        counts: dict[int, int] = {}
        for card in cards:
            if card.year is not None and has(card):
                counts[card.year] = counts.get(card.year, 0) + 1
        m = momentum_score(value, counts)
        return MomentumLite(maturity=str(m.maturity), composite=m.composite, burst=m.burst)

    def _open_problems_for(self, cards: list[PaperCard], row: str, col: str) -> list[str]:
        """Corpus future-work lines that mention either side of the cell."""
        terms = {t for t in (row.lower(), col.lower()) if t}
        out: list[str] = []
        seen: set[str] = set()
        for card in cards:
            for fw in card.future_work:
                low = fw.lower()
                if any(t in low for t in terms):
                    key = fw.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        out.append(fw.strip())
        return out[:6]

    async def research_proposal(
        self,
        row_facet: str,
        col_facet: str,
        row: str,
        col: str,
        *,
        global_count: int = 0,
    ) -> dict[str, object]:
        """Generate a grounded research proposal for one gap-matrix cell."""
        from lattice.landscape.proposal import build_proposal
        from lattice.landscape.signals import DemandScorer

        cards = await self._active_cards()
        row, col = row.lower(), col.lower()
        papers = self._facet_papers(cards, row_facet, col_facet)
        demand = DemandScorer.from_cards(self.chunk_embedder, cards).score(row, col)
        proposal = build_proposal(
            row_facet,
            col_facet,
            row,
            col,
            papers,
            now_year=datetime.now(UTC).year,
            global_count=global_count,
            demand=demand,
            row_momentum=self._facet_momentum(cards, row_facet, row),
            col_momentum=self._facet_momentum(cards, col_facet, col),
            open_problems=self._open_problems_for(cards, row, col),
        )
        return proposal.to_json()

    async def research_opportunities(
        self,
        row_facet: str,
        col_facet: str,
        *,
        limit: int = 5,
        global_counts: dict[tuple[str, str], int] | None = None,
    ) -> dict[str, object]:
        """Rank the corpus's top gaps and draft a full proposal for each.

        Builds the gap matrix, takes the highest-pressure empty/blind-spot cells,
        and generates a grounded proposal per cell (reusing any global counts the
        matrix already fetched, so this costs no extra network calls).
        """
        from lattice.landscape.matrix import PaperFacets, build_gap_matrix, top_gaps
        from lattice.landscape.signals import DemandScorer

        cards = await self._active_cards()
        global_counts = global_counts or {}
        now_year = datetime.now(UTC).year
        facets = [
            PaperFacets(
                paper_id=c.paper_id,
                year=c.year,
                methods=c.normalized_methods,
                datasets=c.normalized_datasets,
                concepts={d.lower() for d in c.domains},
            )
            for c in cards
        ]
        demand = DemandScorer.from_cards(self.chunk_embedder, cards)
        cells = build_gap_matrix(
            facets,
            row_facet,
            col_facet,
            now_year=now_year,
            global_count_fn=lambda r, cv: global_counts.get((r, cv), 0),
            demand_fn=demand.score,
        )
        proposals = []
        for cell in top_gaps(cells, limit=limit):
            proposals.append(
                await self.research_proposal(
                    row_facet, col_facet, cell.row, cell.col, global_count=cell.global_count
                )
            )
        # Strongest opportunities first by the proposal's own confidence.
        proposals.sort(key=lambda p: float(p["confidence"]), reverse=True)  # type: ignore[arg-type]
        return {"row_facet": row_facet, "col_facet": col_facet, "proposals": proposals}

    # ------------------------------------------------------------------ related work / export
    async def _cards_by_community(self) -> dict[str, list[PaperCard]]:
        snapshot = await self.graph_snapshot()
        community_of = {n.id: str(n.community) for n in snapshot.nodes}
        groups: dict[str, list[PaperCard]] = {}
        for card in await self._active_cards():
            groups.setdefault(community_of.get(card.paper_id, "0"), []).append(card)
        return groups

    async def related_work(self) -> dict[str, object]:
        from lattice.rag.related_work import build_related_work

        draft = build_related_work(await self._cards_by_community())
        return {
            "clusters": [cl.to_json() for cl in draft.clusters],
            "markdown": draft.markdown(),
            "bibtex": draft.bibtex(),
        }

    async def export_bibtex(self) -> str:
        from lattice.rag.related_work import corpus_to_bibtex

        return corpus_to_bibtex(await self._active_cards())

    async def export_obsidian(self) -> dict[str, str]:
        from lattice.export.obsidian import card_to_note

        snapshot = await self.graph_snapshot()
        neighbors: dict[str, list[tuple[str, float]]] = {}
        titles = {n.id: n.title for n in snapshot.nodes}
        for e in snapshot.edges:
            neighbors.setdefault(e.source, []).append((e.target, e.weight))
            neighbors.setdefault(e.target, []).append((e.source, e.weight))
        notes: dict[str, str] = {}
        for card in await self._active_cards():
            nbrs = [(titles.get(t, t), w) for t, w in neighbors.get(card.paper_id, [])]
            notes[card.title] = card_to_note(card, nbrs)
        return notes

    # ------------------------------------------------------------------ digest
    async def generate_digest(self, period_label: str | None = None) -> dict[str, object]:
        """Build a delta digest from the current corpus (new papers, edges, movers,
        contradictions). Returns the report JSON + rendered Markdown."""
        from lattice.digest.weekly import (
            Contradiction,
            DigestInput,
            NewPaper,
            build_digest,
            render_markdown,
        )
        from lattice.graph.contradictions import ClaimRelation
        from lattice.landscape.momentum import momentum_score

        cards = await self._active_cards()
        snapshot = await self.graph_snapshot()
        label = period_label or datetime.now(UTC).strftime("%Y-W%V")

        counts: dict[str, dict[int, int]] = {}
        for card in cards:
            if card.year is None:
                continue
            for concept in card.domains:
                counts.setdefault(concept.lower(), {})
                counts[concept.lower()][card.year] = counts[concept.lower()].get(card.year, 0) + 1
        movers = [momentum_score(concept, ys) for concept, ys in counts.items()]

        contradictions = [
            Contradiction(e.source_text, e.source_paper, e.target_text, e.target_paper)
            for e in self._claim_relations
            if e.relation == ClaimRelation.CONTRADICTS
        ]

        report = build_digest(
            DigestInput(
                period_label=label,
                new_papers=[NewPaper(c.paper_id, c.title, c.year) for c in cards],
                new_edges=len(snapshot.edges),
                contradictions=contradictions,
                movers=movers,
            )
        )
        payload: dict[str, object] = {
            "report": report.to_json(),
            "markdown": render_markdown(report),
        }
        return payload

    # ------------------------------------------------------------------ lineage
    async def lineage(self, method: str) -> dict[str, object]:
        from lattice.graph.lineage import LineageNode, build_lineage

        if self.reader is not None:
            return (await self.reader.lineage(self.settings.workspace_id, method)).to_json()

        nodes = [
            LineageNode(
                paper_id=card.paper_id,
                title=card.title,
                year=card.year,
                methods=card.normalized_methods,
            )
            for card in await self._active_cards()
        ]
        return build_lineage(nodes, list(self._cites), method).to_json()

    # ------------------------------------------------------------------ reading queue
    async def reading_queue(self, read_ids: set[str] | None = None) -> list[dict[str, object]]:
        """Rank unread papers by expected information gain given the graph.

        ``read_ids`` are papers already read; the rest are ranked. When empty, all
        papers are candidates and corpus methods are drawn from the whole corpus.
        """
        from lattice.landscape.reading_queue import ReadingCandidate, rank_reading_queue

        read_ids = read_ids or set()
        snapshot = await self.graph_snapshot()
        centrality = {n.id: n.centrality for n in snapshot.nodes}
        neighbors: dict[str, list[tuple[float, float]]] = {}
        for e in snapshot.edges:
            # Count an edge toward a paper only if its other end is already read.
            if e.target in read_ids:
                neighbors.setdefault(e.source, []).append((e.weight, centrality.get(e.target, 0.0)))
            if e.source in read_ids:
                neighbors.setdefault(e.target, []).append((e.weight, centrality.get(e.source, 0.0)))
            if not read_ids:
                neighbors.setdefault(e.source, []).append((e.weight, centrality.get(e.target, 0.0)))
                neighbors.setdefault(e.target, []).append((e.weight, centrality.get(e.source, 0.0)))

        cards = await self._active_cards()
        corpus_methods: set[str] = set()
        for card in cards:
            if not read_ids or card.paper_id in read_ids:
                corpus_methods |= card.normalized_methods

        candidates = [
            ReadingCandidate(
                paper_id=card.paper_id,
                title=card.title,
                methods=card.normalized_methods,
                neighbors=neighbors.get(card.paper_id, []),
            )
            for card in cards
            if card.paper_id not in read_ids
        ]
        return [s.to_json() for s in rank_reading_queue(candidates, corpus_methods)]

    # ------------------------------------------------------------------ selection summary
    async def summarize_papers(self, paper_ids: list[str]) -> dict[str, object]:
        """Synthesize a grounded brief for a hand-selected subset of papers.

        Drives the explorer's lasso-select-to-summarize: given the ids inside a
        selection, return shared methods/datasets/domains, the year span, common
        open problems, any contradictions *within* the selection, a representative
        claim per paper, and a rendered Markdown brief. Fully deterministic and
        offline (no LLM), so it works in demo mode and is grounded by construction.
        """
        from collections import Counter

        from lattice.core.hashing import normalize_text
        from lattice.rag.related_work import bibtex_key

        wanted = list(dict.fromkeys(paper_ids))  # dedupe, preserve order
        # Deliberately all_cards, not _active_cards: an explicitly selected paper
        # (e.g. a superseded preprint opened from its page) must still summarize.
        by_id = {c.paper_id: c for c in await self.cards.all_cards()}
        cards = [by_id[pid] for pid in wanted if pid in by_id]
        if not cards:
            return {
                "count": 0,
                "papers": [],
                "methods": [],
                "datasets": [],
                "domains": [],
                "year_range": None,
                "open_problems": [],
                "contradictions": [],
                "markdown": "_No papers in selection._\n",
            }

        methods: Counter[str] = Counter()
        datasets: Counter[str] = Counter()
        domains: Counter[str] = Counter()
        problems: Counter[str] = Counter()
        problem_canonical: dict[str, str] = {}
        years: list[int] = []
        for c in cards:
            methods.update(m for m in (c.methods_taxonomy + c.methodology.techniques) if m.strip())
            datasets.update(d.name for d in c.datasets if d.name.strip())
            domains.update(d for d in c.domains if d.strip())
            if c.year is not None:
                years.append(c.year)
            for fw in c.future_work:
                key = normalize_text(fw)
                if key:
                    problem_canonical.setdefault(key, fw.strip())
                    problems[key] += 1

        # Contradictions whose *both* endpoints fall inside the selection.
        selected = {c.paper_id for c in cards}
        contradictions = [
            {
                "source_paper": e.source_paper,
                "source_text": e.source_text,
                "target_paper": e.target_paper,
                "target_text": e.target_text,
                "confidence": round(e.confidence, 3),
            }
            for e in await self.get_claim_relations("CONTRADICTS")
            if e.source_paper in selected and e.target_paper in selected
        ]

        papers = [
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "year": c.year,
                "key": bibtex_key(c),
                "representative_claim": c.key_results[0].claim if c.key_results else None,
            }
            for c in sorted(cards, key=lambda c: c.year or 0)
        ]
        top_methods = [m for m, _ in methods.most_common(8)]
        top_datasets = [d for d, _ in datasets.most_common(6)]
        top_domains = [d for d, _ in domains.most_common(4)]
        shared_problems = [
            {"problem": problem_canonical[k], "mentions": n}
            for k, n in problems.most_common(6)
            if n >= 2
        ]
        year_range = [min(years), max(years)] if years else None

        markdown = self._summary_markdown(
            cards,
            top_methods,
            top_datasets,
            top_domains,
            year_range,
            shared_problems,
            contradictions,
        )
        return {
            "count": len(cards),
            "papers": papers,
            "methods": top_methods,
            "datasets": top_datasets,
            "domains": top_domains,
            "year_range": year_range,
            "open_problems": shared_problems,
            "contradictions": contradictions,
            "markdown": markdown,
        }

    @staticmethod
    def _summary_markdown(
        cards: list[PaperCard],
        methods: list[str],
        datasets: list[str],
        domains: list[str],
        year_range: list[int] | None,
        problems: list[dict[str, object]],
        contradictions: list[dict[str, object]],
    ) -> str:
        span = f"{year_range[0]}-{year_range[1]}" if year_range else "n/a"
        lines = [
            f"# Selection brief - {len(cards)} papers",
            "",
            f"**Span:** {span}  -  **Themes:** {', '.join(domains) or 'mixed'}",
            "",
        ]
        if methods:
            lines += [f"**Shared methods:** {', '.join(methods)}", ""]
        if datasets:
            lines += [f"**Shared datasets:** {', '.join(datasets)}", ""]
        if contradictions:
            lines.append("## Tensions within the selection")
            for ct in contradictions:
                lines.append(f"- _{ct['source_text']}_ vs. _{ct['target_text']}_")
            lines.append("")
        if problems:
            lines.append("## Recurring open problems")
            for p in problems:
                lines.append(f"- {p['problem']} ({p['mentions']} papers)")
            lines.append("")
        lines.append("## Papers")
        for c in sorted(cards, key=lambda c: c.year or 0):
            claim = f" — {c.key_results[0].claim}" if c.key_results else ""
            lines.append(f"- **{c.title}** ({c.year or '?'}){claim}")
        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------ explorer
    async def graph_snapshot(self, as_of_year: int | None = None) -> GraphSnapshot:
        """Nodes + edges for the explorer, with quick community + centrality.

        Uses greedy-modularity communities and weighted-degree centrality so the
        demo graph is meaningful without a Neo4j GDS run. Production reads precomputed
        PageRank/Louvain from the graph store via the injected reader.

        When ``as_of_year`` is set, the graph is reconstructed *as of* that year:
        only papers published up to it (and edges whose both endpoints qualify) are
        included, and communities/centrality are recomputed on that subgraph. This
        powers the time-travel replay of how the field evolved.
        """
        if self.reader is not None:
            return await self.reader.snapshot(self.settings.workspace_id, as_of_year)
        cards = {c.paper_id: c for c in await self._active_cards()}
        if as_of_year is not None:
            cards = {
                pid: c for pid, c in cards.items() if c.year is not None and c.year <= as_of_year
            }
        # Deduplicate undirected edges (max weight).
        undirected: dict[tuple[str, str], GraphEdge] = {}
        for src, edges in self._related.items():
            if src not in cards:  # time-travel: source filtered out
                continue
            for e in edges:
                if e.target_id not in cards:
                    continue
                a, b = sorted((src, e.target_id))
                key = (a, b)
                prev = undirected.get(key)
                if prev is None or e.weight > prev.weight:
                    undirected[key] = GraphEdge(
                        source=a,
                        target=b,
                        weight=round(e.weight, 4),
                        components={k: round(v, 4) for k, v in e.result.components.items()},
                    )

        # Real communities via greedy modularity (not connected components, which
        # collapse a connected graph into one blob). Persistent path uses Neo4j GDS.
        from lattice.graph.community import greedy_modularity_communities

        community_ids = greedy_modularity_communities(
            list(cards.keys()),
            [(e.source, e.target, e.weight) for e in undirected.values()],
        )

        degree: dict[str, float] = dict.fromkeys(cards, 0.0)
        for edge in undirected.values():
            degree[edge.source] += edge.weight
            degree[edge.target] += edge.weight
        max_deg = max(degree.values(), default=1.0) or 1.0

        nodes = [
            GraphNode(
                id=pid,
                title=card.title,
                year=card.year,
                community=community_ids[pid],
                centrality=round(degree[pid] / max_deg, 4),
                needs_review=card.needs_review,
            )
            for pid, card in cards.items()
        ]
        return GraphSnapshot(nodes=nodes, edges=list(undirected.values()))

    async def graph_timeline(self) -> dict[str, object]:
        """Year bounds + cumulative corpus growth, for the time-travel slider.

        ``buckets`` is the running paper count up to and including each year, so the
        UI can show how the field accreted and bound the slider to real data.
        """
        years = sorted(c.year for c in await self._active_cards() if c.year is not None)
        if not years:
            return {"min_year": None, "max_year": None, "buckets": []}
        lo, hi = years[0], years[-1]
        buckets = [
            {"year": y, "papers": sum(1 for yr in years if yr <= y)} for y in range(lo, hi + 1)
        ]
        return {"min_year": lo, "max_year": hi, "buckets": buckets}

    async def graph_delta(
        self, since_year: int, until_year: int | None = None
    ) -> dict[str, object]:
        """What changed between two points on the publication-year axis.

        Returns papers and edges present at ``until_year`` (default: now) but not at
        ``since_year`` - i.e. what entered the field in (since, until]. Computed as a
        set difference of two reconstructed snapshots, so it is correct for both the
        in-memory and Neo4j-backed paths without any extra queries.
        """
        before = await self.graph_snapshot(as_of_year=since_year)
        after = await self.graph_snapshot(as_of_year=until_year)
        before_nodes = {n.id for n in before.nodes}
        before_edges = {(e.source, e.target) for e in before.edges}
        new_nodes = [n for n in after.nodes if n.id not in before_nodes]
        new_edges = [e for e in after.edges if (e.source, e.target) not in before_edges]
        new_nodes.sort(key=lambda n: (n.year or 0, n.title))
        new_edges.sort(key=lambda e: e.weight, reverse=True)
        return {
            "since_year": since_year,
            "until_year": until_year,
            "new_papers": [{"paper_id": n.id, "title": n.title, "year": n.year} for n in new_nodes],
            "new_edges": [e.to_json() for e in new_edges],
            "counts": {"papers": len(new_nodes), "edges": len(new_edges)},
        }

    # ------------------------------------------------------------------ helpers
    def _ann_candidates(self, pid: str, new_feat: PaperFeatures) -> list[PaperFeatures]:
        scored: list[tuple[float, PaperFeatures]] = []
        for other_pid, feat in self._features.items():
            if other_pid == pid:
                continue
            sim = cosine(new_feat.specter, feat.specter)
            scored.append((sim or 0.0, feat))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _s, f in scored[: self.settings.similarity.candidate_k]]

    #: Max vectors used for a calibrator refit. All-pairs over the whole corpus is
    #: O(n^2) on EVERY ingest (O(n^3) across a backfill); the p5/p95 percentile
    #: estimates converge long before ~2k pairs, so cap at 64 vectors (2016 pairs).
    _CALIBRATION_VECS = 64

    def _fit_calibrator(self) -> CosineCalibrator:
        vecs = [f.specter for f in self._features.values() if f.specter is not None]
        if len(vecs) < 3:
            return CosineCalibrator()
        if len(vecs) > self._CALIBRATION_VECS:
            # Deterministic, evenly spaced subsample: reproducible for a given
            # corpus, no RNG, and it spans early and late ingests alike.
            step = len(vecs) / self._CALIBRATION_VECS
            vecs = [vecs[int(i * step)] for i in range(self._CALIBRATION_VECS)]
        cosines: list[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = cosine(vecs[i], vecs[j])
                if c is not None:
                    cosines.append(c)
        return CosineCalibrator.fit(cosines)

    def _citation_signals(
        self, new_feat: PaperFeatures, candidates: list[PaperFeatures]
    ) -> dict[str, CitationSignal]:
        out: dict[str, CitationSignal] = {}
        for cand in candidates:
            cand_ext = self._external_ids.get(cand.paper_id, set())
            direct = bool(cand_ext & new_feat.references)
            if direct:
                out[cand.paper_id] = CitationSignal(direct_citation=True)
        return out
