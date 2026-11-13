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

export type AppView = 'overview' | 'predictions' | 'models';

export const VALID_VIEWS: readonly AppView[] = ['overview', 'predictions', 'models'];

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

// ==========================================
// Stage 3 — Model Evaluation & Benchmarks
// ==========================================

export type ModelEvaluationTab = 'phase5_selection' | 'phase4_benchmark' | 'configuration_tuning';

export type Phase4SortMetric = 'endpoint_mae' | 'row_mae' | 'row_rmse' | 'macro_mae';

export interface Phase4BenchmarkRecord {
  dataset: DatasetId;
  model: string;
  row_mae: number;
  row_rmse: number;
  row_r2: number;
  n_observations: number;
  macro_mae: number;
  macro_rmse: number;
  n_engines: number;
  endpoint_mae: number;
  endpoint_rmse: number;
  endpoint_r2: number;
  endpoint_nasa_score: number;
  endpoint_nasa_avg: number;
  mean_signed_error: number;
  n_underpredictions: number;
  n_overpredictions: number;
  runtime: number;
}

export interface Phase5CrossValidationRecord {
  dataset: DatasetId;
  model: string;
  engine_macro_mae: number;
  engine_macro_mae_std: number;
  engine_macro_rmse: number;
  pooled_mae: number;
  pooled_rmse: number;
  mean_signed_error: number;
  nasa_score_sum: number;
  nasa_score_mean: number;
}

export interface Phase5DevelopmentValidationRecord {
  dataset: DatasetId;
  model: string;
  target_cap: number | 'None';
  engine_macro_mae: number;
  engine_macro_rmse: number;
  pooled_mae: number;
  pooled_rmse: number;
  mean_signed_error: number;
  nasa_score_sum: number;
  nasa_score_mean: number;
  n_observations: number;
  n_unique_engines: number;
}

export interface Phase5WindowAblationRecord {
  dataset: DatasetId;
  window_configuration: string;
  target_window: number | 'ALL';
  feature_count: number;
  cv_engine_macro_mae: number;
  cv_pooled_mae: number;
  cv_pooled_rmse: number;
  cv_nasa_score_mean: number;
  val_engine_macro_mae: number;
  val_pooled_mae: number;
  val_pooled_rmse: number;
  val_nasa_score_sum: number;
  val_nasa_score_mean: number;
}

export interface Phase5TargetCapSensitivityRecord {
  dataset: DatasetId;
  target_cap: number | 'None';
  output_constraint: string;
  cv_engine_macro_mae: number;
  cv_pooled_mae: number;
  cv_pooled_rmse: number;
  cv_nasa_score_mean: number;
  val_engine_macro_mae: number;
  val_pooled_mae: number;
  val_pooled_rmse: number;
  val_nasa_score_sum: number;
  val_nasa_score_mean: number;
  n_val_observations: number;
  n_val_engines: number;
}

export interface HyperparameterTrialRecord {
  dataset: DatasetId;
  model_family: string;
  trial_id: number;
  params: Record<string, unknown>;
  params_raw: string;
  engine_macro_mae: number;
  pooled_mae: number;
  pooled_rmse: number;
  nasa_score_mean: number;
  runtime_sec: number;
  status: string;
}


