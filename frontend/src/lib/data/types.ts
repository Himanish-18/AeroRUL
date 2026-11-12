/**
 * AeroRUL — Core Data Contracts and Type Definitions
 *
 * Strict interfaces matching research artifacts:
 * - final_test_metrics.csv
 * - final_test_predictions.csv
 * - final_model_manifest.json
 */

export type DatasetId = 'FD001' | 'FD002' | 'FD003' | 'FD004';

export const VALID_DATASETS: readonly DatasetId[] = ['FD001', 'FD002', 'FD003', 'FD004'];

export type AppView = 'overview' | 'predictions';

export const VALID_VIEWS: readonly AppView[] = ['overview', 'predictions'];

export interface TestMetricRecord {
  dataset: DatasetId;
  selected_model: string;
  selected_feature_configuration: string;
  selected_target_cap: number | 'None';
  endpoint_mae: number;
  endpoint_rmse: number;
  endpoint_r2: number;
  mean_signed_error: number;
  underprediction_count: number;
  overprediction_count: number;
  nasa_score_sum: number;
  nasa_score_mean: number;
  n_official_test_engines: number;
  n_successfully_predicted: number;
}

export interface PredictionRecord {
  dataset: DatasetId;
  unit_id: number;
  last_observed_cycle: number;
  actual_rul_official: number;
  predicted_rul_raw: number;
  predicted_rul_final: number;
  prediction_error: number;
  absolute_error: number;
  selected_model: string;
  selected_feature_configuration: string;
  selected_target_cap: number | 'None';
}

export interface ManifestScores {
  engine_macro_mae: number;
  pooled_mae: number;
  pooled_rmse: number;
  nasa_score_mean?: number;
  nasa_score_sum?: number;
}

export interface DatasetManifestConfig {
  dataset: DatasetId;
  model_family: string;
  hyperparameters: Record<string, unknown>;
  feature_configuration: string;
  target_cap: string;
  output_constraint: string;
  needs_scaling: boolean;
  inner_cv_scores: ManifestScores;
  development_validation_scores: ManifestScores;
  n_features: number;
  feature_generation_mode: string;
  random_seed: number;
  tuning_study_identifier: string;
  selection_metric: string;
}

export interface ModelManifest {
  manifest_version: string;
  frozen_timestamp: string;
  project: string;
  phase: string;
  random_seed: number;
  selected_configurations: Record<DatasetId, DatasetManifestConfig>;
  system_metadata?: {
    python_version?: string;
    package_versions?: Record<string, string>;
    platform?: string;
  };
}

export type SortField =
  | 'unit_id'
  | 'last_observed_cycle'
  | 'actual_rul_official'
  | 'predicted_rul_raw'
  | 'predicted_rul_final'
  | 'prediction_error'
  | 'absolute_error';

export type SortDirection = 'asc' | 'desc';

export interface SortState {
  field: SortField;
  direction: SortDirection;
}

export type ErrorClassificationFilter = 'all' | 'over' | 'under';

export interface PredictionFilterState {
  searchQuery: string;
  errorClassification: ErrorClassificationFilter;
  minAbsoluteError: number;
}

export interface EngineFilterState {
  searchQuery: string;
  errorCategory?: 'all' | 'under' | 'over';
}

export interface PredictionSummaryStats {
  totalEvaluated: number;
  successfullyPredicted: number;
  coveragePercent: string;
  overpredictionCount: number;
  overpredictionPercent: number;
  underpredictionCount: number;
  underpredictionPercent: number;
  exactCount: number;
  exactPercent: number;
  endpointMae: number;
  medianAbsoluteError: number;
  p90AbsoluteError: number;
  meanSignedError: number;
  nasaScoreSum: number;
}

export interface ErrorDistributionBin {
  binStart: number;
  binEnd: number;
  binLabel: string;
  count: number;
  percentage: number;
  isOverprediction: boolean;
  isUnderprediction: boolean;
  isZeroBin: boolean;
}

