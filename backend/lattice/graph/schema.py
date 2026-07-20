"""Neo4j constraints and indexes for the persisted graph schema.

Owning the ontology is the product. Generic frameworks treat all entities the
same; a literature graph has a known, rich ontology and a precise multi-component
similarity function, so the schema is explicit and constrained.
"""

from lattice.graph.store import GraphStore

#: Uniqueness constraints (also create backing indexes). Scoped by workspace_id
#: via composite keys so the graph is multi-tenant ready.
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS "
    "FOR (p:Paper) REQUIRE (p.workspace_id, p.id) IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS "
    "FOR (a:Author) REQUIRE (a.workspace_id, a.key) IS UNIQUE",
    "CREATE CONSTRAINT method_key IF NOT EXISTS "
    "FOR (m:Method) REQUIRE (m.workspace_id, m.key) IS UNIQUE",
    "CREATE CONSTRAINT dataset_key IF NOT EXISTS "
    "FOR (d:Dataset) REQUIRE (d.workspace_id, d.key) IS UNIQUE",
    "CREATE CONSTRAINT concept_key IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE (c.workspace_id, c.key) IS UNIQUE",
    "CREATE CONSTRAINT claim_id IF NOT EXISTS "
    "FOR (cl:Claim) REQUIRE (cl.workspace_id, cl.id) IS UNIQUE",
    "CREATE CONSTRAINT open_problem_id IF NOT EXISTS "
    "FOR (op:OpenProblem) REQUIRE (op.workspace_id, op.id) IS UNIQUE",
]

INDEXES: list[str] = [
    "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
    "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
    "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
]


async def apply_schema(store: GraphStore) -> None:
    """Create all constraints and indexes. Idempotent (IF NOT EXISTS)."""
    for stmt in CONSTRAINTS + INDEXES:
        await store.execute(stmt)
