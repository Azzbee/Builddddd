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
from lattice.db.blobs import BlobStore, InMemoryBlobStore
from lattice.db.cards import CorpusStore
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
)
from lattice.graph.entity_resolution import EntityResolver
from lattice.graph.evolution import (
    CitationSignal,
    EdgeUpdate,
    compute_related_edges,
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
    ParsedDocument,
    SourceType,
)
from lattice.ingestion.pdf_utils import classify_pdf
from lattice.ingestion.pipeline import IngestionPipeline, PipelineContext

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

    # Per-paper feature state for O(k) incremental linking (hydrated from DB in prod).
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
                page=c.page,
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
        self._aspects[pid] = aspects

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

        await self.cards.put_card(card)
        # Persist the source PDF last (after the card exists, for the FK), so the
        # reader can open it and citations can deep-link to a page.
        if ctx.raw_pdf:
            meta = await self.blobs.put(pid, ctx.raw_pdf, content_hash=ctx.job.content_hash)
            ctx.extra["pdf_pages"] = meta.pages
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
            cluster_open_problems,
            consolidate_known_knowns,
            cross_community_transfer,
        )

        cards = await self.cards.all_cards()
        now_year = datetime.now(UTC).year

        # Known knowns: group identical claims across papers.
        from lattice.core.hashing import normalize_text

        grouped: dict[str, list[SupportingPaper]] = {}
        canonical: dict[str, str] = {}
        for card in cards:
            authors = frozenset(a.normalized_name for a in card.authors)
            method = card.methods_taxonomy[0] if card.methods_taxonomy else None
            dataset = card.datasets[0].name if card.datasets else None
            for r in card.key_results:
                key = normalize_text(r.claim)
                if not key:
                    continue
                canonical.setdefault(key, r.claim)
                grouped.setdefault(key, []).append(
                    SupportingPaper(card.paper_id, authors, method, dataset, card.year)
                )
        contradictions = await self.get_claim_relations("CONTRADICTS")
        contradicted = {normalize_text(e.source_text) for e in contradictions} | {
            normalize_text(e.target_text) for e in contradictions
        }
        findings = consolidate_known_knowns(grouped, contradicted)
        known_knowns = [
            {
                "claim": canonical.get(normalize_text(f.claim), f.claim),
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
            (card.methods_taxonomy[0] if card.methods_taxonomy else "method", card.paper_id, self._aspects[card.paper_id]["methodology"])
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

    # ------------------------------------------------------------------ related work / export
    async def _cards_by_community(self) -> dict[str, list[PaperCard]]:
        snapshot = await self.graph_snapshot()
        community_of = {n.id: str(n.community) for n in snapshot.nodes}
        groups: dict[str, list[PaperCard]] = {}
        for card in await self.cards.all_cards():
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

        return corpus_to_bibtex(await self.cards.all_cards())

    async def export_obsidian(self) -> dict[str, str]:
        from lattice.export.obsidian import card_to_note

        snapshot = await self.graph_snapshot()
        neighbors: dict[str, list[tuple[str, float]]] = {}
        titles = {n.id: n.title for n in snapshot.nodes}
        for e in snapshot.edges:
            neighbors.setdefault(e.source, []).append((e.target, e.weight))
            neighbors.setdefault(e.target, []).append((e.source, e.weight))
        notes: dict[str, str] = {}
        for card in await self.cards.all_cards():
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

        cards = await self.cards.all_cards()
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
        payload: dict[str, object] = {"report": report.to_json(), "markdown": render_markdown(report)}
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
            for card in await self.cards.all_cards()
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

        cards = await self.cards.all_cards()
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
        by_id = {c.paper_id: c for c in await self.cards.all_cards()}
        cards = [by_id[pid] for pid in wanted if pid in by_id]
        if not cards:
            return {
                "count": 0, "papers": [], "methods": [], "datasets": [], "domains": [],
                "year_range": None, "open_problems": [], "contradictions": [],
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
                "source_paper": e.source_paper, "source_text": e.source_text,
                "target_paper": e.target_paper, "target_text": e.target_text,
                "confidence": round(e.confidence, 3),
            }
            for e in await self.get_claim_relations("CONTRADICTS")
            if e.source_paper in selected and e.target_paper in selected
        ]

        papers = [
            {
                "paper_id": c.paper_id, "title": c.title, "year": c.year,
                "key": bibtex_key(c),
                "representative_claim": c.key_results[0].claim if c.key_results else None,
            }
            for c in sorted(cards, key=lambda c: (c.year or 0))
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
            cards, top_methods, top_datasets, top_domains, year_range,
            shared_problems, contradictions,
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
        for c in sorted(cards, key=lambda c: (c.year or 0)):
            claim = f" — {c.key_results[0].claim}" if c.key_results else ""
            lines.append(f"- **{c.title}** ({c.year or '?'}){claim}")
        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------ explorer
    async def graph_snapshot(self) -> GraphSnapshot:
        """Nodes + edges for the explorer, with quick community + centrality.

        Uses union-find communities and weighted-degree centrality so the demo
        graph is meaningful without a Neo4j GDS run. Production reads precomputed
        PageRank/Louvain from the graph store via the injected reader.
        """
        if self.reader is not None:
            return await self.reader.snapshot(self.settings.workspace_id)
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
