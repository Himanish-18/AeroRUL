/**
 * AeroRUL — Data Parsing and Schema Validation
 *
 * Validates and transforms raw CSV text and JSON into strictly typed records.
 * Throws explicit descriptive errors on missing columns, NaN values, corrupt numbers,
 * unrecognized statuses, or duplicate logical records.
 */

import Papa from 'papaparse';
import {
  DatasetId,
  VALID_DATASETS,
  TestMetricRecord,
  PredictionRecord,
  ModelManifest,
  Phase4BenchmarkRecord,
  Phase5CrossValidationRecord,
  Phase5DevelopmentValidationRecord,
  Phase5WindowAblationRecord,
  Phase5TargetCapSensitivityRecord,
  HyperparameterTrialRecord,
  FeatureImportanceRecord,
  SensorImportanceRecord,
  FeatureFamilyRecord,
  SensorTrajectoryRecord,
  FeatureImportanceManifest,
  SensorTrajectoriesManifest,
} from './types';

export const VALID_OPTUNA_STATUSES: readonly string[] = [
  'COMPLETE',
  'PRUNED',
  'FAIL',
  'WAITING',
  'RUNNING',
];

export const VALID_MODEL_FAMILIES: readonly string[] = [
  'LightGBM',
  'XGBoost',
];

export function validateRequiredHeaders(
  fields: string[] | undefined,
  required: readonly string[],
  fileLabel: string
): void {
  if (!fields || fields.length === 0) {
    throw new Error(`CSV header row missing or empty in ${fileLabel}`);
  }
  const missing = required.filter((req) => !fields.includes(req));
  if (missing.length > 0) {
    throw new Error(`Missing required headers in ${fileLabel}: ${missing.join(', ')}`);
  }
}

export function parseStrictFloat(value: unknown, fieldName: string, rowIdx: number): number {
  if (value === null || value === undefined) {
    throw new Error(`Missing float value at row ${rowIdx} for field "${fieldName}"`);
  }
  if (typeof value === 'boolean') {
    throw new Error(`Boolean value invalid for float field "${fieldName}" at row ${rowIdx}`);
  }
  const str = String(value).trim();
  if (str === '') {
    throw new Error(`Empty float value at row ${rowIdx} for field "${fieldName}"`);
  }
  const num = Number(str);
  if (!Number.isFinite(num)) {
    throw new Error(`Invalid non-finite float at row ${rowIdx} for field "${fieldName}": ${value}`);
  }
  return num;
}

export function parseStrictPositiveInt(value: unknown, fieldName: string, rowIdx: number): number {
  if (value === null || value === undefined) {
    throw new Error(`Missing integer value at row ${rowIdx} for field "${fieldName}"`);
  }
  if (typeof value === 'boolean') {
    throw new Error(`Boolean value invalid for integer field "${fieldName}" at row ${rowIdx}`);
  }
  const str = String(value).trim();
  if (str === '') {
    throw new Error(`Empty integer value at row ${rowIdx} for field "${fieldName}"`);
  }
  const num = Number(str);
  if (!Number.isInteger(num) || num <= 0) {
    throw new Error(`Invalid positive integer at row ${rowIdx} for field "${fieldName}": ${value}`);
  }
  return num;
}

export function parseStrictNonNegativeInt(value: unknown, fieldName: string, rowIdx: number): number {
  if (value === null || value === undefined) {
    throw new Error(`Missing integer value at row ${rowIdx} for field "${fieldName}"`);
  }
  if (typeof value === 'boolean') {
    throw new Error(`Boolean value invalid for integer field "${fieldName}" at row ${rowIdx}`);
  }
  const str = String(value).trim();
  if (str === '') {
    throw new Error(`Empty integer value at row ${rowIdx} for field "${fieldName}"`);
  }
  const num = Number(str);
  if (!Number.isInteger(num) || num < 0) {
    throw new Error(`Invalid non-negative integer at row ${rowIdx} for field "${fieldName}": ${value}`);
  }
  return num;
}

export function parseStrictTargetCap(value: unknown, rowIdx: number): number | 'None' {
  if (value === null || value === undefined) {
    throw new Error(`Missing target_cap at row ${rowIdx}`);
  }
  const str = String(value).trim();
  if (str === 'None' || str === 'none') {
    return 'None';
  }
  if (str === '') {
    throw new Error(`Empty target_cap at row ${rowIdx}`);
  }
  const num = Number(str);
  if (!Number.isFinite(num) || num <= 0) {
    throw new Error(`Invalid target_cap at row ${rowIdx}: "${value}". Expected "None" or positive number`);
  }
  return num;
}

export function parseStrictTargetWindow(value: unknown, rowIdx: number): number | 'ALL' {
  if (value === null || value === undefined) {
    throw new Error(`Missing target_window at row ${rowIdx}`);
  }
  const str = String(value).trim();
  if (str === 'ALL' || str === 'all') {
    return 'ALL';
  }
  if (str === '') {
    throw new Error(`Empty target_window at row ${rowIdx}`);
  }
  const num = Number(str);
  if (!Number.isInteger(num) || num <= 0) {
    throw new Error(`Invalid target_window at row ${rowIdx}: "${value}". Expected "ALL" or positive integer`);
  }
  return num;
}

export function validateOptunaStatus(status: unknown, rowIdx: number): string {
  const str = String(status || '').trim();
  if (!VALID_OPTUNA_STATUSES.includes(str)) {
    throw new Error(`Invalid Optuna trial status at row ${rowIdx}: "${status}". Expected one of: ${VALID_OPTUNA_STATUSES.join(', ')}`);
  }
  return str;
}

export function validateModelFamily(family: unknown, rowIdx: number): string {
  const str = String(family || '').trim();
  if (!VALID_MODEL_FAMILIES.includes(str)) {
    throw new Error(`Invalid model family at row ${rowIdx}: "${family}". Expected one of: ${VALID_MODEL_FAMILIES.join(', ')}`);
  }
  return str;
}

export function parseDatasetId(value: unknown, rowIdx: number): DatasetId {
  const ds = String(value).trim() as DatasetId;
  if (!VALID_DATASETS.includes(ds)) {
    throw new Error(`Invalid dataset identifier at row ${rowIdx}: "${value}". Expected one of: ${VALID_DATASETS.join(', ')}`);
  }
  return ds;
}

// ==========================================
// CSV Parsers
// ==========================================

const REQUIRED_METRICS_HEADERS = [
  'dataset',
  'selected_model',
  'selected_feature_configuration',
  'selected_target_cap',
  'endpoint_mae',
  'endpoint_rmse',
  'endpoint_r2',
  'mean_signed_error',
  'underprediction_count',
  'overprediction_count',
  'nasa_score_sum',
  'nasa_score_mean',
  'n_official_test_engines',
  'n_successfully_predicted',
] as const;

export function parseMetricsCsv(csvText: string): TestMetricRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Metrics CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in metrics: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Metrics CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_METRICS_HEADERS, 'final_test_metrics.csv');

  const seenDatasets = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    if (seenDatasets.has(dataset)) {
      throw new Error(`Duplicate metrics record for dataset "${dataset}" at row ${idx}`);
    }
    seenDatasets.add(dataset);

    const selected_model = String(row.selected_model || '').trim();
    const selected_feature_configuration = String(row.selected_feature_configuration || '').trim();

    if (!selected_model) throw new Error(`Missing selected_model at row ${idx}`);
    if (!selected_feature_configuration) throw new Error(`Missing selected_feature_configuration at row ${idx}`);

    return {
      dataset,
      selected_model,
      selected_feature_configuration,
      selected_target_cap: parseStrictTargetCap(row.selected_target_cap, idx),
      endpoint_mae: parseStrictFloat(row.endpoint_mae, 'endpoint_mae', idx),
      endpoint_rmse: parseStrictFloat(row.endpoint_rmse, 'endpoint_rmse', idx),
      endpoint_r2: parseStrictFloat(row.endpoint_r2, 'endpoint_r2', idx),
      mean_signed_error: parseStrictFloat(row.mean_signed_error, 'mean_signed_error', idx),
      underprediction_count: parseStrictNonNegativeInt(row.underprediction_count, 'underprediction_count', idx),
      overprediction_count: parseStrictNonNegativeInt(row.overprediction_count, 'overprediction_count', idx),
      nasa_score_sum: parseStrictFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseStrictFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      n_official_test_engines: parseStrictPositiveInt(row.n_official_test_engines, 'n_official_test_engines', idx),
      n_successfully_predicted: parseStrictPositiveInt(row.n_successfully_predicted, 'n_successfully_predicted', idx),
    };
  });
}

const REQUIRED_PREDICTIONS_HEADERS = [
  'dataset',
  'unit_id',
  'last_observed_cycle',
  'actual_rul_official',
  'predicted_rul_raw',
  'predicted_rul_final',
  'prediction_error',
  'absolute_error',
  'selected_model',
  'selected_feature_configuration',
  'selected_target_cap',
] as const;

export function parsePredictionsCsv(csvText: string): PredictionRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Predictions CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in predictions: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Predictions CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PREDICTIONS_HEADERS, 'final_test_predictions.csv');

  const seenUnitKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const unit_id = parseStrictPositiveInt(row.unit_id, 'unit_id', idx);
    const unitKey = `${dataset}:${unit_id}`;
    if (seenUnitKeys.has(unitKey)) {
      throw new Error(`Duplicate prediction record for unit "${unit_id}" in dataset "${dataset}" at row ${idx}`);
    }
    seenUnitKeys.add(unitKey);

    const last_observed_cycle = parseStrictPositiveInt(row.last_observed_cycle, 'last_observed_cycle', idx);

    return {
      dataset,
      unit_id,
      last_observed_cycle,
      actual_rul_official: parseStrictFloat(row.actual_rul_official, 'actual_rul_official', idx),
      predicted_rul_raw: parseStrictFloat(row.predicted_rul_raw, 'predicted_rul_raw', idx),
      predicted_rul_final: parseStrictFloat(row.predicted_rul_final, 'predicted_rul_final', idx),
      prediction_error: parseStrictFloat(row.prediction_error, 'prediction_error', idx),
      absolute_error: parseStrictFloat(row.absolute_error, 'absolute_error', idx),
      selected_model: String(row.selected_model || '').trim(),
      selected_feature_configuration: String(row.selected_feature_configuration || '').trim(),
      selected_target_cap: parseStrictTargetCap(row.selected_target_cap, idx),
    };
  });
}

export function parseModelManifest(jsonObj: unknown): ModelManifest {
  if (!jsonObj || typeof jsonObj !== 'object') {
    throw new Error('Manifest JSON is invalid or not an object');
  }

  const manifest = jsonObj as ModelManifest;
  if (!manifest.manifest_version || !manifest.frozen_timestamp || !manifest.selected_configurations) {
    throw new Error('Manifest missing top-level keys (manifest_version, frozen_timestamp, or selected_configurations)');
  }

  for (const ds of VALID_DATASETS) {
    const config = manifest.selected_configurations[ds];
    if (!config) {
      throw new Error(`Manifest missing configuration for dataset: ${ds}`);
    }
    if (!config.model_family || config.n_features === undefined || !config.feature_configuration) {
      throw new Error(`Incomplete configuration for dataset ${ds} in manifest`);
    }
  }

  return manifest;
}

const REQUIRED_PHASE4_HEADERS = [
  'dataset',
  'model',
  'row_mae',
  'row_rmse',
  'row_r2',
  'n_observations',
  'macro_mae',
  'macro_rmse',
  'n_engines',
  'endpoint_mae',
  'endpoint_rmse',
  'endpoint_r2',
  'endpoint_nasa_score',
  'endpoint_nasa_avg',
  'mean_signed_error',
  'n_underpredictions',
  'n_overpredictions',
  'runtime',
] as const;

export function parsePhase4BenchmarkCsv(csvText: string): Phase4BenchmarkRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Phase 4 benchmark CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in Phase 4 benchmark: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Phase 4 benchmark CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PHASE4_HEADERS, 'model_benchmark_metrics.csv');

  const seenModelKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 4 benchmark`);

    const recordKey = `${dataset}:${model}`;
    if (seenModelKeys.has(recordKey)) {
      throw new Error(`Duplicate Phase 4 benchmark record for dataset "${dataset}", model "${model}" at row ${idx}`);
    }
    seenModelKeys.add(recordKey);

    return {
      dataset,
      model,
      row_mae: parseStrictFloat(row.row_mae, 'row_mae', idx),
      row_rmse: parseStrictFloat(row.row_rmse, 'row_rmse', idx),
      row_r2: parseStrictFloat(row.row_r2, 'row_r2', idx),
      n_observations: parseStrictPositiveInt(row.n_observations, 'n_observations', idx),
      macro_mae: parseStrictFloat(row.macro_mae, 'macro_mae', idx),
      macro_rmse: parseStrictFloat(row.macro_rmse, 'macro_rmse', idx),
      n_engines: parseStrictPositiveInt(row.n_engines, 'n_engines', idx),
      endpoint_mae: parseStrictFloat(row.endpoint_mae, 'endpoint_mae', idx),
      endpoint_rmse: parseStrictFloat(row.endpoint_rmse, 'endpoint_rmse', idx),
      endpoint_r2: parseStrictFloat(row.endpoint_r2, 'endpoint_r2', idx),
      endpoint_nasa_score: parseStrictFloat(row.endpoint_nasa_score, 'endpoint_nasa_score', idx),
      endpoint_nasa_avg: parseStrictFloat(row.endpoint_nasa_avg, 'endpoint_nasa_avg', idx),
      mean_signed_error: parseStrictFloat(row.mean_signed_error, 'mean_signed_error', idx),
      n_underpredictions: parseStrictNonNegativeInt(row.n_underpredictions, 'n_underpredictions', idx),
      n_overpredictions: parseStrictNonNegativeInt(row.n_overpredictions, 'n_overpredictions', idx),
      runtime: parseStrictFloat(row.runtime, 'runtime', idx),
    };
  });
}

const REQUIRED_PHASE5_CV_HEADERS = [
  'dataset',
  'model',
  'engine_macro_mae',
  'engine_macro_mae_std',
  'engine_macro_rmse',
  'pooled_mae',
  'pooled_rmse',
  'mean_signed_error',
  'nasa_score_sum',
  'nasa_score_mean',
] as const;

export function parsePhase5CrossValidationCsv(csvText: string): Phase5CrossValidationRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Phase 5 cross validation CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in Phase 5 cross validation: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Phase 5 cross validation CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PHASE5_CV_HEADERS, 'cross_validation_metrics.csv');

  const seenModelKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 5 cross validation`);

    const recordKey = `${dataset}:${model}`;
    if (seenModelKeys.has(recordKey)) {
      throw new Error(`Duplicate Phase 5 cross validation record for dataset "${dataset}", model "${model}" at row ${idx}`);
    }
    seenModelKeys.add(recordKey);

    return {
      dataset,
      model,
      engine_macro_mae: parseStrictFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      engine_macro_mae_std: parseStrictFloat(row.engine_macro_mae_std, 'engine_macro_mae_std', idx),
      engine_macro_rmse: parseStrictFloat(row.engine_macro_rmse, 'engine_macro_rmse', idx),
      pooled_mae: parseStrictFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseStrictFloat(row.pooled_rmse, 'pooled_rmse', idx),
      mean_signed_error: parseStrictFloat(row.mean_signed_error, 'mean_signed_error', idx),
      nasa_score_sum: parseStrictFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseStrictFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
    };
  });
}

const REQUIRED_PHASE5_DEVVAL_HEADERS = [
  'dataset',
  'model',
  'target_cap',
  'engine_macro_mae',
  'engine_macro_rmse',
  'pooled_mae',
  'pooled_rmse',
  'mean_signed_error',
  'nasa_score_sum',
  'nasa_score_mean',
  'n_observations',
  'n_unique_engines',
] as const;

export function parsePhase5DevelopmentValidationCsv(csvText: string): Phase5DevelopmentValidationRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Phase 5 development validation CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in Phase 5 development validation: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Phase 5 development validation CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PHASE5_DEVVAL_HEADERS, 'development_validation_metrics.csv');

  const seenModelCapKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 5 development validation`);

    const target_cap = parseStrictTargetCap(row.target_cap, idx);
    const recordKey = `${dataset}:${model}:${target_cap}`;
    if (seenModelCapKeys.has(recordKey)) {
      throw new Error(`Duplicate development validation record for dataset "${dataset}", model "${model}", target_cap "${target_cap}" at row ${idx}`);
    }
    seenModelCapKeys.add(recordKey);

    return {
      dataset,
      model,
      target_cap,
      engine_macro_mae: parseStrictFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      engine_macro_rmse: parseStrictFloat(row.engine_macro_rmse, 'engine_macro_rmse', idx),
      pooled_mae: parseStrictFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseStrictFloat(row.pooled_rmse, 'pooled_rmse', idx),
      mean_signed_error: parseStrictFloat(row.mean_signed_error, 'mean_signed_error', idx),
      nasa_score_sum: parseStrictFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseStrictFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      n_observations: parseStrictPositiveInt(row.n_observations, 'n_observations', idx),
      n_unique_engines: parseStrictPositiveInt(row.n_unique_engines, 'n_unique_engines', idx),
    };
  });
}

const REQUIRED_PHASE5_WINDOW_HEADERS = [
  'dataset',
  'window_configuration',
  'target_window',
  'feature_count',
  'cv_engine_macro_mae',
  'cv_pooled_mae',
  'cv_pooled_rmse',
  'cv_nasa_score_mean',
  'val_engine_macro_mae',
  'val_pooled_mae',
  'val_pooled_rmse',
  'val_nasa_score_sum',
  'val_nasa_score_mean',
] as const;

export function parsePhase5WindowAblationCsv(csvText: string): Phase5WindowAblationRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Phase 5 window ablation CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in Phase 5 window ablation: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Phase 5 window ablation CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PHASE5_WINDOW_HEADERS, 'window_ablation_metrics.csv');

  const seenWindowKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const window_configuration = String(row.window_configuration || '').trim();
    if (!window_configuration) throw new Error(`Missing window_configuration at row ${idx}`);

    const recordKey = `${dataset}:${window_configuration}`;
    if (seenWindowKeys.has(recordKey)) {
      throw new Error(`Duplicate window ablation record for dataset "${dataset}", window_configuration "${window_configuration}" at row ${idx}`);
    }
    seenWindowKeys.add(recordKey);

    return {
      dataset,
      window_configuration,
      target_window: parseStrictTargetWindow(row.target_window, idx),
      feature_count: parseStrictPositiveInt(row.feature_count, 'feature_count', idx),
      cv_engine_macro_mae: parseStrictFloat(row.cv_engine_macro_mae, 'cv_engine_macro_mae', idx),
      cv_pooled_mae: parseStrictFloat(row.cv_pooled_mae, 'cv_pooled_mae', idx),
      cv_pooled_rmse: parseStrictFloat(row.cv_pooled_rmse, 'cv_pooled_rmse', idx),
      cv_nasa_score_mean: parseStrictFloat(row.cv_nasa_score_mean, 'cv_nasa_score_mean', idx),
      val_engine_macro_mae: parseStrictFloat(row.val_engine_macro_mae, 'val_engine_macro_mae', idx),
      val_pooled_mae: parseStrictFloat(row.val_pooled_mae, 'val_pooled_mae', idx),
      val_pooled_rmse: parseStrictFloat(row.val_pooled_rmse, 'val_pooled_rmse', idx),
      val_nasa_score_sum: parseStrictFloat(row.val_nasa_score_sum, 'val_nasa_score_sum', idx),
      val_nasa_score_mean: parseStrictFloat(row.val_nasa_score_mean, 'val_nasa_score_mean', idx),
    };
  });
}

const REQUIRED_PHASE5_TARGET_CAP_HEADERS = [
  'dataset',
  'target_cap',
  'output_constraint',
  'cv_engine_macro_mae',
  'cv_pooled_mae',
  'cv_pooled_rmse',
  'cv_nasa_score_mean',
  'val_engine_macro_mae',
  'val_pooled_mae',
  'val_pooled_rmse',
  'val_nasa_score_sum',
  'val_nasa_score_mean',
  'n_val_observations',
  'n_val_engines',
] as const;

export function parsePhase5TargetCapSensitivityCsv(csvText: string): Phase5TargetCapSensitivityRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Phase 5 target cap sensitivity CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in Phase 5 target cap sensitivity: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Phase 5 target cap sensitivity CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_PHASE5_TARGET_CAP_HEADERS, 'target_cap_sensitivity.csv');

  const seenCapKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const target_cap = parseStrictTargetCap(row.target_cap, idx);
    const recordKey = `${dataset}:${target_cap}`;
    if (seenCapKeys.has(recordKey)) {
      throw new Error(`Duplicate target cap sensitivity record for dataset "${dataset}", target_cap "${target_cap}" at row ${idx}`);
    }
    seenCapKeys.add(recordKey);

    const output_constraint = String(row.output_constraint || '').trim();
    if (!output_constraint) throw new Error(`Missing output_constraint at row ${idx}`);

    return {
      dataset,
      target_cap,
      output_constraint,
      cv_engine_macro_mae: parseStrictFloat(row.cv_engine_macro_mae, 'cv_engine_macro_mae', idx),
      cv_pooled_mae: parseStrictFloat(row.cv_pooled_mae, 'cv_pooled_mae', idx),
      cv_pooled_rmse: parseStrictFloat(row.cv_pooled_rmse, 'cv_pooled_rmse', idx),
      cv_nasa_score_mean: parseStrictFloat(row.cv_nasa_score_mean, 'cv_nasa_score_mean', idx),
      val_engine_macro_mae: parseStrictFloat(row.val_engine_macro_mae, 'val_engine_macro_mae', idx),
      val_pooled_mae: parseStrictFloat(row.val_pooled_mae, 'val_pooled_mae', idx),
      val_pooled_rmse: parseStrictFloat(row.val_pooled_rmse, 'val_pooled_rmse', idx),
      val_nasa_score_sum: parseStrictFloat(row.val_nasa_score_sum, 'val_nasa_score_sum', idx),
      val_nasa_score_mean: parseStrictFloat(row.val_nasa_score_mean, 'val_nasa_score_mean', idx),
      n_val_observations: parseStrictPositiveInt(row.n_val_observations, 'n_val_observations', idx),
      n_val_engines: parseStrictPositiveInt(row.n_val_engines, 'n_val_engines', idx),
    };
  });
}

const REQUIRED_HYPERPARAMETER_HEADERS = [
  'dataset',
  'model_family',
  'trial_id',
  'params',
  'engine_macro_mae',
  'pooled_mae',
  'pooled_rmse',
  'nasa_score_mean',
  'runtime_sec',
  'status',
] as const;

export function parseHyperparameterSearchCsv(csvText: string): HyperparameterTrialRecord[] {
  if (!csvText || !csvText.trim()) {
    throw new Error('Hyperparameter search CSV is empty or has no data rows');
  }

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in hyperparameter search: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Hyperparameter search CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_HYPERPARAMETER_HEADERS, 'hyperparameter_search_results.csv');

  const seenTrialKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model_family = validateModelFamily(row.model_family, idx);
    const trial_id = parseStrictNonNegativeInt(row.trial_id, 'trial_id', idx);

    const recordKey = `${dataset}:${model_family}:${trial_id}`;
    if (seenTrialKeys.has(recordKey)) {
      throw new Error(`Duplicate hyperparameter trial record for dataset "${dataset}", model_family "${model_family}", trial_id ${trial_id} at row ${idx}`);
    }
    seenTrialKeys.add(recordKey);

    const params_raw = String(row.params || '').trim();
    if (!params_raw) throw new Error(`Missing params JSON string at row ${idx}`);

    let params: Record<string, unknown>;
    try {
      params = JSON.parse(params_raw);
      if (typeof params !== 'object' || params === null || Array.isArray(params)) {
        throw new Error('params is not a JSON object');
      }
    } catch (e) {
      throw new Error(`Invalid JSON in params at row ${idx}: ${e instanceof Error ? e.message : String(e)}`);
    }

    const status = validateOptunaStatus(row.status, idx);

    return {
      dataset,
      model_family,
      trial_id,
      params,
      params_raw,
      engine_macro_mae: parseStrictFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      pooled_mae: parseStrictFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseStrictFloat(row.pooled_rmse, 'pooled_rmse', idx),
      nasa_score_mean: parseStrictFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      runtime_sec: parseStrictFloat(row.runtime_sec, 'runtime_sec', idx),
      status,
    };
  });
}

// ---------------------------------------------------------------------------
// Stage 4: Feature Importance & Degradation Dynamics Parsers
// ---------------------------------------------------------------------------

export const REQUIRED_FEATURE_IMPORTANCE_HEADERS: readonly string[] = [
  'dataset',
  'feature_name',
  'sensor',
  'statistic',
  'window_or_lag',
  'description',
  'gain_importance',
  'split_importance',
  'gain_importance_pct',
  'split_importance_pct',
  'gain_rank',
  'split_rank',
  'correlation_pearson',
  'correlation_spearman',
  'abs_pearson',
  'abs_spearman',
];

export const REQUIRED_SENSOR_IMPORTANCE_HEADERS: readonly string[] = [
  'dataset',
  'sensor',
  'gain_importance',
  'split_importance',
  'gain_importance_pct',
  'split_importance_pct',
  'gain_rank',
  'split_rank',
  'feature_count',
  'top_feature',
];

export const REQUIRED_FEATURE_FAMILY_HEADERS: readonly string[] = [
  'dataset',
  'family_type',
  'family_key',
  'gain_importance',
  'split_importance',
  'gain_importance_pct',
  'split_importance_pct',
  'feature_count',
  'rank_in_family',
];

export const REQUIRED_SENSOR_TRAJECTORIES_HEADERS: readonly string[] = [
  'dataset',
  'unit_id',
  'cycle',
  'sensor',
  'sensor_value',
];

export function parseOptionalFloat(val: unknown, fieldName: string, rowIdx: number): number | null {
  if (val === null || val === undefined) return null;
  const str = String(val).trim();
  if (str === '' || str.toLowerCase() === 'nan' || str.toLowerCase() === 'null') return null;
  return parseStrictFloat(str, fieldName, rowIdx);
}

export function parseFeatureImportanceCsv(
  csvText: string,
  expectedDataset?: DatasetId
): FeatureImportanceRecord[] {
  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: 'greedy',
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in feature importance: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Feature importance CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_FEATURE_IMPORTANCE_HEADERS, 'feature_importance.csv');

  const seenFeatureKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    if (expectedDataset && dataset !== expectedDataset) {
      throw new Error(`Row ${idx} dataset "${dataset}" does not match expected "${expectedDataset}"`);
    }

    const feature_name = String(row.feature_name || '').trim();
    if (!feature_name) {
      throw new Error(`Missing feature_name at row ${idx}`);
    }

    const recordKey = `${dataset}:${feature_name}`;
    if (seenFeatureKeys.has(recordKey)) {
      throw new Error(`Duplicate feature importance record for dataset "${dataset}", feature "${feature_name}" at row ${idx}`);
    }
    seenFeatureKeys.add(recordKey);

    const gain_importance = parseStrictFloat(row.gain_importance, 'gain_importance', idx);
    if (gain_importance < 0) {
      throw new Error(`gain_importance cannot be negative at row ${idx}: ${gain_importance}`);
    }

    const split_importance = parseStrictNonNegativeInt(row.split_importance, 'split_importance', idx);
    const gain_importance_pct = parseStrictFloat(row.gain_importance_pct, 'gain_importance_pct', idx);
    const split_importance_pct = parseStrictFloat(row.split_importance_pct, 'split_importance_pct', idx);
    const gain_rank = parseStrictPositiveInt(row.gain_rank, 'gain_rank', idx);
    const split_rank = parseStrictPositiveInt(row.split_rank, 'split_rank', idx);

    const correlation_pearson = parseOptionalFloat(row.correlation_pearson, 'correlation_pearson', idx);
    const correlation_spearman = parseOptionalFloat(row.correlation_spearman, 'correlation_spearman', idx);
    const abs_pearson = parseOptionalFloat(row.abs_pearson, 'abs_pearson', idx);
    const abs_spearman = parseOptionalFloat(row.abs_spearman, 'abs_spearman', idx);

    return {
      dataset,
      feature_name,
      sensor: String(row.sensor || '').trim(),
      statistic: String(row.statistic || '').trim(),
      window_or_lag: String(row.window_or_lag || '').trim(),
      description: String(row.description || '').trim(),
      gain_importance,
      split_importance,
      gain_importance_pct,
      split_importance_pct,
      gain_rank,
      split_rank,
      correlation_pearson,
      correlation_spearman,
      abs_pearson,
      abs_spearman,
    };
  });
}

export function parseSensorImportanceCsv(
  csvText: string,
  expectedDataset?: DatasetId
): SensorImportanceRecord[] {
  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: 'greedy',
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in sensor importance: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Sensor importance CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_SENSOR_IMPORTANCE_HEADERS, 'sensor_importance.csv');

  const seenSensorKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    if (expectedDataset && dataset !== expectedDataset) {
      throw new Error(`Row ${idx} dataset "${dataset}" does not match expected "${expectedDataset}"`);
    }

    const sensor = String(row.sensor || '').trim();
    if (!sensor) {
      throw new Error(`Missing sensor at row ${idx}`);
    }

    const recordKey = `${dataset}:${sensor}`;
    if (seenSensorKeys.has(recordKey)) {
      throw new Error(`Duplicate sensor record for dataset "${dataset}", sensor "${sensor}" at row ${idx}`);
    }
    seenSensorKeys.add(recordKey);

    return {
      dataset,
      sensor,
      gain_importance: parseStrictFloat(row.gain_importance, 'gain_importance', idx),
      split_importance: parseStrictNonNegativeInt(row.split_importance, 'split_importance', idx),
      gain_importance_pct: parseStrictFloat(row.gain_importance_pct, 'gain_importance_pct', idx),
      split_importance_pct: parseStrictFloat(row.split_importance_pct, 'split_importance_pct', idx),
      gain_rank: parseStrictPositiveInt(row.gain_rank, 'gain_rank', idx),
      split_rank: parseStrictPositiveInt(row.split_rank, 'split_rank', idx),
      feature_count: parseStrictPositiveInt(row.feature_count, 'feature_count', idx),
      top_feature: String(row.top_feature || '').trim(),
    };
  });
}

export function parseFeatureFamilyCsv(csvText: string): FeatureFamilyRecord[] {
  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: 'greedy',
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in feature family importance: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Feature family importance CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_FEATURE_FAMILY_HEADERS, 'feature_family_importance.csv');

  const seenFamilyKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const rawFamType = String(row.family_type || '').trim();
    if (!['sensor', 'statistic', 'window'].includes(rawFamType)) {
      throw new Error(`Invalid family_type "${rawFamType}" at row ${idx}`);
    }
    const family_type = rawFamType as 'sensor' | 'statistic' | 'window';
    const family_key = String(row.family_key || '').trim();
    if (!family_key) {
      throw new Error(`Missing family_key at row ${idx}`);
    }

    const recordKey = `${dataset}:${family_type}:${family_key}`;
    if (seenFamilyKeys.has(recordKey)) {
      throw new Error(`Duplicate family record for key "${recordKey}" at row ${idx}`);
    }
    seenFamilyKeys.add(recordKey);

    return {
      dataset,
      family_type,
      family_key,
      gain_importance: parseStrictFloat(row.gain_importance, 'gain_importance', idx),
      split_importance: parseStrictNonNegativeInt(row.split_importance, 'split_importance', idx),
      gain_importance_pct: parseStrictFloat(row.gain_importance_pct, 'gain_importance_pct', idx),
      split_importance_pct: parseStrictFloat(row.split_importance_pct, 'split_importance_pct', idx),
      feature_count: parseStrictPositiveInt(row.feature_count, 'feature_count', idx),
      rank_in_family: parseStrictPositiveInt(row.rank_in_family, 'rank_in_family', idx),
    };
  });
}

export function parseSensorTrajectoriesCsv(
  csvText: string,
  expectedDataset?: DatasetId
): SensorTrajectoryRecord[] {
  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: 'greedy',
  });

  if (parsed.errors.length > 0) {
    throw new Error(`CSV parse error in sensor trajectories: ${parsed.errors[0].message}`);
  }

  const rows = parsed.data;
  if (rows.length === 0) {
    throw new Error('Sensor trajectories CSV is empty or has no data rows');
  }

  validateRequiredHeaders(parsed.meta.fields, REQUIRED_SENSOR_TRAJECTORIES_HEADERS, 'sensor_trajectories.csv');

  const seenTrajectoryKeys = new Set<string>();

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    if (expectedDataset && dataset !== expectedDataset) {
      throw new Error(`Row ${idx} dataset "${dataset}" does not match expected "${expectedDataset}"`);
    }

    const unit_id = parseStrictPositiveInt(row.unit_id, 'unit_id', idx);
    const cycle = parseStrictPositiveInt(row.cycle, 'cycle', idx);
    const sensor = String(row.sensor || '').trim();
    if (!sensor) {
      throw new Error(`Missing sensor identifier at row ${idx}`);
    }

    const recordKey = `${dataset}:${unit_id}:${cycle}:${sensor}`;
    if (seenTrajectoryKeys.has(recordKey)) {
      throw new Error(`Duplicate trajectory point for key "${recordKey}" at row ${idx}`);
    }
    seenTrajectoryKeys.add(recordKey);

    const sensor_value = parseStrictFloat(row.sensor_value, 'sensor_value', idx);

    return {
      dataset,
      unit_id,
      cycle,
      sensor,
      sensor_value,
    };
  });
}

export function parseFeatureImportanceManifest(jsonObj: unknown): FeatureImportanceManifest {
  if (typeof jsonObj !== 'object' || jsonObj === null || Array.isArray(jsonObj)) {
    throw new Error('Feature importance manifest is not a valid JSON object');
  }
  const obj = jsonObj as Record<string, unknown>;
  if (!obj.manifest_version || !obj.generated_at || !obj.datasets) {
    throw new Error('Feature importance manifest is missing required top-level keys');
  }
  return obj as unknown as FeatureImportanceManifest;
}

export function parseSensorTrajectoriesManifest(jsonObj: unknown): SensorTrajectoriesManifest {
  if (typeof jsonObj !== 'object' || jsonObj === null || Array.isArray(jsonObj)) {
    throw new Error('Sensor trajectories manifest is not a valid JSON object');
  }
  const obj = jsonObj as Record<string, unknown>;
  if (!obj.manifest_version || !obj.generated_at || !obj.subsets) {
    throw new Error('Sensor trajectories manifest is missing required top-level keys');
  }
  return obj as unknown as SensorTrajectoriesManifest;
}

