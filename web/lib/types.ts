export interface PaperSummary {
  paper_id: string;
  title: string;
  year: number | null;
  authors: string[];
  paper_type: string;
  confidence: number;
  needs_review: boolean;
  superseded_by?: string | null;
}

export interface ResultItem {
  claim: string;
  metric: string | null;
  value: string | null;
  baseline_comparison: string | null;
  evidence_location: string;
}

export interface DatasetRef {
  name: string;
  source: string | null;
  size: string | null;
  is_public: boolean | null;
  url: string | null;
}

export interface Methodology {
  approach_summary: string;
  method_family: string[];
  techniques: string[];
  evaluation_protocol: string | null;
  baselines: string[];
}

export interface PaperCard {
  paper_id: string;
  title: string;
  authors: { name: string }[];
  year: number | null;
  venue: string | null;
  doi: string | null;
  superseded_by?: string | null;
  arxiv_id: string | null;
  abstract: string | null;
  problem_statement: string;
  research_questions: string[];
  methodology: Methodology;
  datasets: DatasetRef[];
  key_results: ResultItem[];
  limitations: string[];
  contributions: string[];
  future_work: string[];
  paper_type: string;
  domains: string[];
  methods_taxonomy: string[];
  confidence: number;
  needs_review: boolean;
  review_reasons: string[];
}

export interface GraphNode {
  id: string;
  title: string;
  year: number | null;
  community: number;
  centrality: number;
  needs_review: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  components: Record<string, number>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphTimeline {
  min_year: number | null;
  max_year: number | null;
  buckets: { year: number; papers: number }[];
}

export interface GraphDelta {
  since_year: number;
  until_year: number | null;
  new_papers: { paper_id: string; title: string; year: number | null }[];
  new_edges: GraphEdge[];
  counts: { papers: number; edges: number };
}

export interface Citation {
  marker: number;
  paper_id: string;
  title: string;
  section: string | null;
  evidence_location: string | null;
  snippet: string | null;
  page: number | null;
}

export interface PdfMeta {
  paper_id: string;
  available: boolean;
  pages: number;
  size: number;
}

export interface SelectionSummary {
  count: number;
  papers: {
    paper_id: string;
    title: string;
    year: number | null;
    key: string;
    representative_claim: string | null;
  }[];
  methods: string[];
  datasets: string[];
  domains: string[];
  year_range: [number, number] | null;
  open_problems: { problem: string; mentions: number }[];
  contradictions: {
    source_paper: string;
    source_text: string;
    target_paper: string;
    target_text: string;
    confidence: number;
  }[];
  markdown: string;
}

export interface AgentAnswer {
  answer: string;
  citations: Citation[];
  query_class: string;
  confidence: number;
  abstained: boolean;
  cost_usd: number;
}

export interface MatrixCell {
  row: string;
  col: string;
  paper_count: number;
  paper_ids: string[];
  latest_year: number | null;
  state: string;
  global_count: number;
  gap_score: number;
  components: { feasibility: number; adjacency_pressure: number; demand_signal: number };
}

export interface ProposalEvidence {
  paper_id: string;
  title: string;
  year: number | null;
  note: string;
}

export interface ResearchProposal {
  row_facet: string;
  col_facet: string;
  row: string;
  col: string;
  state: string;
  novelty: string;
  gap_score: number;
  components: Record<string, number>;
  global_count: number;
  demand: number;
  thesis: string;
  why_now: string;
  row_track_record: ProposalEvidence[];
  col_track_record: ProposalEvidence[];
  method_to_borrow: ProposalEvidence | null;
  recommended_baselines: string[];
  prior_art: ProposalEvidence[];
  risks: string[];
  open_problems: string[];
  confidence: number;
  markdown: string;
}

export interface IngestJob {
  job_id: string;
  paper_id: string | null;
  stage: string;
  status: string;
  error_code: string | null;
  progress?: number;
}
