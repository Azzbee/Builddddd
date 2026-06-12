export interface PaperSummary {
  paper_id: string;
  title: string;
  year: number | null;
  authors: string[];
  paper_type: string;
  confidence: number;
  needs_review: boolean;
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

export interface Citation {
  marker: number;
  paper_id: string;
  title: string;
  section: string | null;
  evidence_location: string | null;
  snippet: string | null;
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

export interface IngestJob {
  job_id: string;
  paper_id: string | null;
  stage: string;
  status: string;
  error_code: string | null;
  progress?: number;
}
