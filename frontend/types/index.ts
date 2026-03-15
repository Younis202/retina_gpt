export interface QualityResult {
  score: number;
  adequate: boolean;
}

export interface DRResult {
  grade: number;
  label: string;
  confidence: number;
  probabilities: number[];
  refer: boolean;
}

export interface AMDResult {
  stage: number;
  label: string;
  confidence: number;
}

export interface GlaucomaResult {
  suspect: boolean;
  cup_disc_ratio: number;
  confidence: number;
}

export interface LesionItem {
  present: boolean;
  probability: number;
}

export interface ReportResult {
  structured_findings: string;
  recommendation: string;
  full_text: string;
}

export interface ExplainabilityResult {
  gradcam_image: string | null;
  attention_image: string | null;
  explanation_panel: string | null;
}

export interface SegmentationResult {
  vessel_mask: string | null;
  optic_disc_mask: string | null;
}

export interface FullAnalysisResponse {
  request_id: string;
  image_id: string;
  quality: QualityResult;
  dr_grading: DRResult;
  amd: AMDResult;
  glaucoma: GlaucomaResult;
  lesions: Record<string, LesionItem>;
  report: ReportResult;
  explainability: ExplainabilityResult;
  segmentation: SegmentationResult;
  inference_time_ms: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  version: string;
  uptime_seconds: number;
  capabilities: Record<string, boolean>;
}

export interface CasesStatsResponse {
  total_cases: number;
  today: number;
  this_week: number;
  referable_cases: number;
  dr_grade_distribution: Record<string, number>;
}

export interface Case {
  id: string;
  patient_id: string;
  created_at: string;
  image_name: string;
  dr_grade: number;
  dr_label: string;
  dr_confidence: number;
  dr_refer: number;
  quality_score: number;
  quality_adequate: number;
  full_result?: string;
  status: string;
}

export interface CasesResponse {
  total: number;
  cases: Case[];
}

export interface SearchResultItem {
  rank: number;
  image_id: string;
  score: number;
  distance: number;
  dr_grade: number | null;
  dr_label: string | null;
  dataset: string | null;
  image_path: string | null;
}

export interface SearchResponse {
  query_id: string;
  num_results: number;
  search_time_ms: number;
  index_size: number;
  results: SearchResultItem[];
}
