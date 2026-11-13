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

export type AppView = 'overview' | 'predictions' | 'models' | 'features';

export const VALID_VIEWS: readonly AppView[] = ['overview', 'predictions', 'models', 'features'];

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

export interface RankComparisonItem {
  model: string;
  cvRank: number | null;
  cvMae: number | null;
  devRank: number | null;
  devCommonRank: number | null;
  devMae: number | null;
  rankDelta: number | null;
  isReversal: boolean;
  isChallengerVariant: boolean;
}

export interface HyperparameterTrialSummary {
  totalStoredTrials: number;
  completedTrials: number;
  bestEngineMacroMae: number;
  medianEngineMacroMae: number;
  bestPooledMae: number;
  bestPooledMaeTrialId: number;
  bestPooledRmse: number;
  bestPooledRmseTrialId: number;
  bestTrialId: number;
  runtimeSecBestTrial: number;
}

// ---------------------------------------------------------------------------
// Stage 4: Feature Intelligence & Degradation Dynamics
// ---------------------------------------------------------------------------

export interface FeatureImportanceRecord {
  dataset: DatasetId;
  feature_name: string;
  sensor: string;
  statistic: string;
  window_or_lag: string;
  description: string;
  gain_importance: number;
  split_importance: number;
  gain_importance_pct: number;
  split_importance_pct: number;
  gain_rank: number;
  split_rank: number;
  correlation_pearson: number | null;
  correlation_spearman: number | null;
  abs_pearson: number | null;
  abs_spearman: number | null;
}

export interface SensorImportanceRecord {
  dataset: DatasetId;
  sensor: string;
  gain_importance: number;
  split_importance: number;
  gain_importance_pct: number;
  split_importance_pct: number;
  gain_rank: number;
  split_rank: number;
  feature_count: number;
  top_feature: string;
}

export interface FeatureFamilyRecord {
  dataset: DatasetId;
  family_type: 'sensor' | 'statistic' | 'window';
  family_key: string;
  gain_importance: number;
  split_importance: number;
  gain_importance_pct: number;
  split_importance_pct: number;
  feature_count: number;
  rank_in_family: number;
}

export interface SensorTrajectoryRecord {
  dataset: DatasetId;
  unit_id: number;
  cycle: number;
  sensor: string;
  sensor_value: number;
}

export interface FeatureManifestEntry {
  dataset: DatasetId;
  source_model_path: string;
  model_file_sha256: string;
  model_family: string;
  feature_configuration: string;
  n_features: number;
  total_gain_importance: number;
  total_split_importance: number;
  importance_methods: string[];
  normalization_method: string;
  top_feature_by_gain: string;
  top_feature_gain_pct: number;
  top_sensor_by_gain: string;
  top_sensor_gain_pct: number;
}

export interface FeatureImportanceManifest {
  manifest_version: string;
  generated_at: string;
  datasets: Record<DatasetId, FeatureManifestEntry>;
  interpretability_notice: string;
}

export interface SensorMetadata {
  name: string;
  unit: string;
  type: string;
}

export interface SensorTrajectoriesManifest {
  manifest_version: string;
  generated_at: string;
  sampling_strategy: string;
  sensor_specifications: Record<string, SensorMetadata>;
  subsets: Record<DatasetId, {
    dataset: DatasetId;
    default_unit_id: number;
    selected_units: number[];
    unit_metadata: Record<string, {
      unit_id: number;
      lifetime_cycles: number;
      sampled_points: number;
      is_fleet_standard: boolean;
    }>;
    available_sensors: string[];
    total_sampled_cycles: number;
    total_measurement_records: number;
  }>;
}



