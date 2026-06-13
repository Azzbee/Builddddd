"""Ingestion service: wires every stage into the resumable pipeline.

This is where M1-M4 come together. Each stage handler is a method that reads and
populates the shared PipelineContext. All collaborators are injected (parser, LLM,
enrichment, embedders, stores, graph writer), so the entire pipeline runs and is
tested offline with in-memory stores and a scripted model. Linking is the final,
transactional stage: a crash before it leaves no partial paper in the graph.
"""

from __future__ import annotations

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
from lattice.db.cards import InMemoryCardStore
from lattice.db.vector import InMemoryVectorStore, VectorRecord
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
)
from lattice.graph.entity_resolution import EntityResolver
from lattice.graph.evolution import (
    CitationSignal,
    EdgeUpdate,
    compute_related_edges,
    diff_audit,
)
from lattice.graph.similarity import CosineCalibrator, PaperFeatures, cosine
from lattice.graph.store import FakeGraphStore, GraphStore
from lattice.graph.writer import GraphWriter
from lattice.ingestion.chunker import chunk_document
from lattice.ingestion.dedup import PaperIdentity
from lattice.ingestion.models import (
    IngestJob,
    JobStage,
    ParsedDocument,
    SourceType,
)
from lattice.ingestion.pdf_utils import classify_pdf
from lattice.ingestion.pipeline import IngestionPipeline, PipelineContext

log = get_logger("ingestion.service")


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    components: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "components": self.components,
        }


@dataclass
class GraphNode:
    id: str
    title: str
    year: int | None
    community: int
    centrality: float
    needs_review: bool

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "community": self.community,
            "centrality": self.centrality,
            "needs_review": self.needs_review,
        }


@dataclass
class GraphSnapshot:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


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
    vectors: InMemoryVectorStore
    cards: InMemoryCardStore
    graph: GraphStore = field(default_factory=FakeGraphStore)
    enricher: Enricher | None = None
    specter: Specter2Embedder = field(default_factory=lambda: Specter2Embedder(dim=768))
    chunk_embedder: ChunkEmbedder = field(default_factory=lambda: ChunkEmbedder(dim=1024))
    aspect_embedder: AspectEmbedder = field(default_factory=lambda: AspectEmbedder(dim=1024))

    # Per-paper feature state for O(k) incremental linking (hydrated from DB in prod).
    _features: dict[str, PaperFeatures] = field(default_factory=dict)
    _external_ids: dict[str, set[str]] = field(default_factory=dict)
    #: In-memory edge mirror so the explorer works without Neo4j (demo/dev mode).
    _related: dict[str, list[EdgeUpdate]] = field(default_factory=dict)
    #: In-memory claim-relation mirror (contradictions/convergence) for the API.
    _claim_relations: list[ClaimEdge] = field(default_factory=list)
    method_resolver: EntityResolver | None = None
    dataset_resolver: EntityResolver | None = None
    #: Text extractor for PDF sanity checks (defaults to pypdf via classify_pdf).
    text_extractor: Callable[[bytes], str] | None = None

    def __post_init__(self) -> None:
        ws = self.settings.workspace_id
        self.method_resolver = self.method_resolver or EntityResolver("method", ws)
        self.dataset_resolver = self.dataset_resolver or EntityResolver("dataset", ws)
        self._writer = GraphWriter(self.graph, ws)

    # ------------------------------------------------------------------ public
    async def ingest_pdf(self, source_ref: str, pdf_bytes: bytes) -> IngestJob:
        job = IngestJob(
            job_id=stable_id("job", source_ref, content_hash(pdf_bytes)),
            workspace_id=self.settings.workspace_id,
            source_type=SourceType.FILE,
            source_ref=source_ref,
            content_hash=content_hash(pdf_bytes),
        )
        ctx = PipelineContext(job=job, raw_pdf=pdf_bytes)
        ctx.extra["cost"] = CostTracker(
            per_job_cap=self.settings.cost.per_job_usd_cap,
            daily_cap=self.settings.cost.daily_usd_cap,
        )
        pipeline = IngestionPipeline(
            {
                JobStage.PARSING: self._parse,
                JobStage.EXTRACTING: self._extract,
                JobStage.ENRICHING: self._enrich,
                JobStage.EMBEDDING: self._embed,
                JobStage.LINKING: self._link,
            }
        )
        return await pipeline.run(ctx)

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
        index.add(PaperIdentity(paper_id, doc.title, doc.authors, doc.doi, doc.arxiv_id, ctx.job.content_hash))
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

    async def _embed(self, ctx: PipelineContext) -> None:
        assert ctx.card is not None and ctx.document is not None
        card = ctx.card
        pid = card.paper_id

        precomputed = ctx.extra.get("specter_embedding")
        paper_emb = self.specter.embed(
            card.title, card.abstract, precomputed=precomputed  # type: ignore[arg-type]
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
            )
            for c in chunks
        ]
        await self.vectors.upsert_chunks(records)
        ctx.chunks = chunks

        import numpy as np

        refs_value = ctx.extra.get("reference_ids")
        refs = set(refs_value) if isinstance(refs_value, list | set) else set()
        self._features[pid] = PaperFeatures(
            paper_id=pid,
            specter=np.asarray(paper_emb.vector, dtype=float),
            methodology_embedding=np.asarray(aspects["methodology"], dtype=float),
            methods=card.normalized_methods,
            datasets=card.normalized_datasets,
            references=refs,
            ingested_at=datetime.now(UTC),
        )
        self._external_ids[pid] = _self_external_ids(card)

    async def _link(self, ctx: PipelineContext) -> None:
        assert ctx.card is not None and ctx.job.paper_id is not None
        card = ctx.card
        pid = card.paper_id
        new_feat = self._features[pid]

        # Candidate generation: top-k by SPECTER cosine among existing papers.
        candidates = self._ann_candidates(pid, new_feat)
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
        for method in card.methods_taxonomy + card.methodology.techniques:
            res = self.method_resolver.resolve(method)  # type: ignore[union-attr]
            await self._writer.upsert_method(res.key, res.name, pid)
        for ds in card.datasets:
            res = self.dataset_resolver.resolve(ds.name)  # type: ignore[union-attr]
            await self._writer.upsert_dataset(res.key, ds.name, pid)
        for concept in card.domains:
            await self._writer.upsert_concept(stable_id(self.settings.workspace_id, "concept", concept), concept, pid)
        for result in card.key_results:
            await self._writer.upsert_claim(pid, result.claim, result.evidence_location)

        existing = await self._writer.existing_related_weights(pid)
        audit = diff_audit(edges, existing)
        for edge in edges:
            await self._writer.upsert_related_edge(edge)
        for sig_pid, sig in citations.items():
            if sig.direct_citation:
                await self._writer.upsert_cites(pid, sig_pid)
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

        await self.cards.put_card(card)
        self._related[pid] = edges
        ctx.extra["edges_written"] = len(edges)

    # ------------------------------------------------------------------ contradictions
    async def analyze_relations(self, judge: NLIJudge | None = None) -> list[ClaimEdge]:
        """Detect SUPPORTS/CONTRADICTS/EXTENDS relations across the corpus' claims.

        Groups every card's key results into ClaimRecords keyed by concept, runs the
        judge over same-concept cross-paper pairs, persists the resulting edges to
        the graph, and mirrors them in memory for the API. Defaults to the offline
        heuristic judge; pass an LLM judge in production.
        """
        judge = judge or HeuristicNLIJudge()
        ws = self.settings.workspace_id
        claims: list[ClaimRecord] = []
        for card in await self.cards.all_cards():
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

    def claim_relations(self) -> list[ClaimEdge]:
        return self._claim_relations

    # ------------------------------------------------------------------ explorer
    async def graph_snapshot(self) -> GraphSnapshot:
        """Nodes + edges for the explorer, with quick community + centrality.

        Uses union-find communities and weighted-degree centrality so the demo
        graph is meaningful without a Neo4j GDS run. Production reads precomputed
        PageRank/Louvain from the graph store.
        """
        cards = {c.paper_id: c for c in await self.cards.all_cards()}
        # Deduplicate undirected edges (max weight).
        undirected: dict[tuple[str, str], GraphEdge] = {}
        for src, edges in self._related.items():
            for e in edges:
                if e.target_id not in cards:
                    continue
                a, b = sorted((src, e.target_id))
                key = (a, b)
                prev = undirected.get(key)
                if prev is None or e.weight > prev.weight:
                    undirected[key] = GraphEdge(
                        source=a, target=b, weight=round(e.weight, 4),
                        components={k: round(v, 4) for k, v in e.result.components.items()},
                    )

        # Union-find communities.
        parent: dict[str, str] = {pid: pid for pid in cards}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for edge in undirected.values():
            ra, rb = find(edge.source), find(edge.target)
            if ra != rb:
                parent[ra] = rb

        labels: dict[str, int] = {}
        community_ids: dict[str, int] = {}
        for pid in cards:
            root = find(pid)
            community_ids[pid] = labels.setdefault(root, len(labels))

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

    def _fit_calibrator(self) -> CosineCalibrator:
        vecs = [f.specter for f in self._features.values() if f.specter is not None]
        if len(vecs) < 3:
            return CosineCalibrator()
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


def _self_external_ids(card: PaperCard) -> set[str]:
    ids: set[str] = set()
    if card.doi:
        ids.add(f"DOI:{card.doi}")
    if card.arxiv_id:
        ids.add(card.arxiv_id)
    if card.s2_paper_id:
        ids.add(card.s2_paper_id)
    return ids
