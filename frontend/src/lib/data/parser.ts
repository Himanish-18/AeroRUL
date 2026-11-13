/**
 * AeroRUL — Data Parsing and Schema Validation
 *
 * Validates and transforms raw CSV text and JSON into strictly typed records.
 * Throws explicit descriptive errors on missing columns, NaN values, or corrupted types.
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
} from './types';

function parseTargetCap(value: unknown): number | 'None' {
  if (value === null || value === undefined) return 'None';
  const str = String(value).trim();
  if (str === 'None' || str === '' || str.toLowerCase() === 'null') {
    return 'None';
  }
  const num = Number(str);
  if (Number.isNaN(num)) {
    return 'None';
  }
  return num;
}

function parseFiniteFloat(value: unknown, fieldName: string, rowIdx: number): number {
  const num = typeof value === 'number' ? value : parseFloat(String(value));
  if (!Number.isFinite(num)) {
    throw new Error(`Invalid non-finite float at row ${rowIdx} for field "${fieldName}": ${value}`);
  }
  return num;
}

function parsePositiveInt(value: unknown, fieldName: string, rowIdx: number): number {
  const num = typeof value === 'number' ? value : parseInt(String(value), 10);
  if (!Number.isInteger(num) || num < 0) {
    throw new Error(`Invalid positive integer at row ${rowIdx} for field "${fieldName}": ${value}`);
  }
  return num;
}

function parseDatasetId(value: unknown, rowIdx: number): DatasetId {
  const ds = String(value).trim() as DatasetId;
  if (!VALID_DATASETS.includes(ds)) {
    throw new Error(`Invalid dataset identifier at row ${rowIdx}: "${value}". Expected one of: ${VALID_DATASETS.join(', ')}`);
  }
  return ds;
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const selected_model = String(row.selected_model || '').trim();
    const selected_feature_configuration = String(row.selected_feature_configuration || '').trim();

    if (!selected_model) throw new Error(`Missing selected_model at row ${idx}`);
    if (!selected_feature_configuration) throw new Error(`Missing selected_feature_configuration at row ${idx}`);

    return {
      dataset,
      selected_model,
      selected_feature_configuration,
      selected_target_cap: parseTargetCap(row.selected_target_cap),
      endpoint_mae: parseFiniteFloat(row.endpoint_mae, 'endpoint_mae', idx),
      endpoint_rmse: parseFiniteFloat(row.endpoint_rmse, 'endpoint_rmse', idx),
      endpoint_r2: parseFiniteFloat(row.endpoint_r2, 'endpoint_r2', idx),
      mean_signed_error: parseFiniteFloat(row.mean_signed_error, 'mean_signed_error', idx),
      underprediction_count: parsePositiveInt(row.underprediction_count, 'underprediction_count', idx),
      overprediction_count: parsePositiveInt(row.overprediction_count, 'overprediction_count', idx),
      nasa_score_sum: parseFiniteFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseFiniteFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      n_official_test_engines: parsePositiveInt(row.n_official_test_engines, 'n_official_test_engines', idx),
      n_successfully_predicted: parsePositiveInt(row.n_successfully_predicted, 'n_successfully_predicted', idx),
    };
  });
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const unit_id = parsePositiveInt(row.unit_id, 'unit_id', idx);
    const last_observed_cycle = parsePositiveInt(row.last_observed_cycle, 'last_observed_cycle', idx);

    return {
      dataset,
      unit_id,
      last_observed_cycle,
      actual_rul_official: parseFiniteFloat(row.actual_rul_official, 'actual_rul_official', idx),
      predicted_rul_raw: parseFiniteFloat(row.predicted_rul_raw, 'predicted_rul_raw', idx),
      predicted_rul_final: parseFiniteFloat(row.predicted_rul_final, 'predicted_rul_final', idx),
      prediction_error: parseFiniteFloat(row.prediction_error, 'prediction_error', idx),
      absolute_error: parseFiniteFloat(row.absolute_error, 'absolute_error', idx),
      selected_model: String(row.selected_model || '').trim(),
      selected_feature_configuration: String(row.selected_feature_configuration || '').trim(),
      selected_target_cap: parseTargetCap(row.selected_target_cap),
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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 4 benchmark`);

    return {
      dataset,
      model,
      row_mae: parseFiniteFloat(row.row_mae, 'row_mae', idx),
      row_rmse: parseFiniteFloat(row.row_rmse, 'row_rmse', idx),
      row_r2: parseFiniteFloat(row.row_r2, 'row_r2', idx),
      n_observations: parsePositiveInt(row.n_observations, 'n_observations', idx),
      macro_mae: parseFiniteFloat(row.macro_mae, 'macro_mae', idx),
      macro_rmse: parseFiniteFloat(row.macro_rmse, 'macro_rmse', idx),
      n_engines: parsePositiveInt(row.n_engines, 'n_engines', idx),
      endpoint_mae: parseFiniteFloat(row.endpoint_mae, 'endpoint_mae', idx),
      endpoint_rmse: parseFiniteFloat(row.endpoint_rmse, 'endpoint_rmse', idx),
      endpoint_r2: parseFiniteFloat(row.endpoint_r2, 'endpoint_r2', idx),
      endpoint_nasa_score: parseFiniteFloat(row.endpoint_nasa_score, 'endpoint_nasa_score', idx),
      endpoint_nasa_avg: parseFiniteFloat(row.endpoint_nasa_avg, 'endpoint_nasa_avg', idx),
      mean_signed_error: parseFiniteFloat(row.mean_signed_error, 'mean_signed_error', idx),
      n_underpredictions: parsePositiveInt(row.n_underpredictions, 'n_underpredictions', idx),
      n_overpredictions: parsePositiveInt(row.n_overpredictions, 'n_overpredictions', idx),
      runtime: parseFiniteFloat(row.runtime, 'runtime', idx),
    };
  });
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 5 cross validation`);

    return {
      dataset,
      model,
      engine_macro_mae: parseFiniteFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      engine_macro_mae_std: parseFiniteFloat(row.engine_macro_mae_std, 'engine_macro_mae_std', idx),
      engine_macro_rmse: parseFiniteFloat(row.engine_macro_rmse, 'engine_macro_rmse', idx),
      pooled_mae: parseFiniteFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseFiniteFloat(row.pooled_rmse, 'pooled_rmse', idx),
      mean_signed_error: parseFiniteFloat(row.mean_signed_error, 'mean_signed_error', idx),
      nasa_score_sum: parseFiniteFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseFiniteFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
    };
  });
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model = String(row.model || '').trim();
    if (!model) throw new Error(`Missing model at row ${idx} in Phase 5 development validation`);

    return {
      dataset,
      model,
      target_cap: parseTargetCap(row.target_cap),
      engine_macro_mae: parseFiniteFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      engine_macro_rmse: parseFiniteFloat(row.engine_macro_rmse, 'engine_macro_rmse', idx),
      pooled_mae: parseFiniteFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseFiniteFloat(row.pooled_rmse, 'pooled_rmse', idx),
      mean_signed_error: parseFiniteFloat(row.mean_signed_error, 'mean_signed_error', idx),
      nasa_score_sum: parseFiniteFloat(row.nasa_score_sum, 'nasa_score_sum', idx),
      nasa_score_mean: parseFiniteFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      n_observations: parsePositiveInt(row.n_observations, 'n_observations', idx),
      n_unique_engines: parsePositiveInt(row.n_unique_engines, 'n_unique_engines', idx),
    };
  });
}

function parseTargetWindow(value: unknown): number | 'ALL' {
  const str = String(value || '').trim();
  if (str === 'ALL' || str.toLowerCase() === 'all') {
    return 'ALL';
  }
  const num = parseInt(str, 10);
  if (!Number.isInteger(num)) {
    return 'ALL';
  }
  return num;
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const window_configuration = String(row.window_configuration || '').trim();
    if (!window_configuration) throw new Error(`Missing window_configuration at row ${idx}`);

    return {
      dataset,
      window_configuration,
      target_window: parseTargetWindow(row.target_window),
      feature_count: parsePositiveInt(row.feature_count, 'feature_count', idx),
      cv_engine_macro_mae: parseFiniteFloat(row.cv_engine_macro_mae, 'cv_engine_macro_mae', idx),
      cv_pooled_mae: parseFiniteFloat(row.cv_pooled_mae, 'cv_pooled_mae', idx),
      cv_pooled_rmse: parseFiniteFloat(row.cv_pooled_rmse, 'cv_pooled_rmse', idx),
      cv_nasa_score_mean: parseFiniteFloat(row.cv_nasa_score_mean, 'cv_nasa_score_mean', idx),
      val_engine_macro_mae: parseFiniteFloat(row.val_engine_macro_mae, 'val_engine_macro_mae', idx),
      val_pooled_mae: parseFiniteFloat(row.val_pooled_mae, 'val_pooled_mae', idx),
      val_pooled_rmse: parseFiniteFloat(row.val_pooled_rmse, 'val_pooled_rmse', idx),
      val_nasa_score_sum: parseFiniteFloat(row.val_nasa_score_sum, 'val_nasa_score_sum', idx),
      val_nasa_score_mean: parseFiniteFloat(row.val_nasa_score_mean, 'val_nasa_score_mean', idx),
    };
  });
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const output_constraint = String(row.output_constraint || '').trim();
    if (!output_constraint) throw new Error(`Missing output_constraint at row ${idx}`);

    return {
      dataset,
      target_cap: parseTargetCap(row.target_cap),
      output_constraint,
      cv_engine_macro_mae: parseFiniteFloat(row.cv_engine_macro_mae, 'cv_engine_macro_mae', idx),
      cv_pooled_mae: parseFiniteFloat(row.cv_pooled_mae, 'cv_pooled_mae', idx),
      cv_pooled_rmse: parseFiniteFloat(row.cv_pooled_rmse, 'cv_pooled_rmse', idx),
      cv_nasa_score_mean: parseFiniteFloat(row.cv_nasa_score_mean, 'cv_nasa_score_mean', idx),
      val_engine_macro_mae: parseFiniteFloat(row.val_engine_macro_mae, 'val_engine_macro_mae', idx),
      val_pooled_mae: parseFiniteFloat(row.val_pooled_mae, 'val_pooled_mae', idx),
      val_pooled_rmse: parseFiniteFloat(row.val_pooled_rmse, 'val_pooled_rmse', idx),
      val_nasa_score_sum: parseFiniteFloat(row.val_nasa_score_sum, 'val_nasa_score_sum', idx),
      val_nasa_score_mean: parseFiniteFloat(row.val_nasa_score_mean, 'val_nasa_score_mean', idx),
      n_val_observations: parsePositiveInt(row.n_val_observations, 'n_val_observations', idx),
      n_val_engines: parsePositiveInt(row.n_val_engines, 'n_val_engines', idx),
    };
  });
}

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

  return rows.map((row, idx) => {
    const dataset = parseDatasetId(row.dataset, idx);
    const model_family = String(row.model_family || '').trim();
    if (!model_family) throw new Error(`Missing model_family at row ${idx}`);

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

    return {
      dataset,
      model_family,
      trial_id: parsePositiveInt(row.trial_id, 'trial_id', idx),
      params,
      params_raw,
      engine_macro_mae: parseFiniteFloat(row.engine_macro_mae, 'engine_macro_mae', idx),
      pooled_mae: parseFiniteFloat(row.pooled_mae, 'pooled_mae', idx),
      pooled_rmse: parseFiniteFloat(row.pooled_rmse, 'pooled_rmse', idx),
      nasa_score_mean: parseFiniteFloat(row.nasa_score_mean, 'nasa_score_mean', idx),
      runtime_sec: parseFiniteFloat(row.runtime_sec, 'runtime_sec', idx),
      status: String(row.status || '').trim(),
    };
  });
}
