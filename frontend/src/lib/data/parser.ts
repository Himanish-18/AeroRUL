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
