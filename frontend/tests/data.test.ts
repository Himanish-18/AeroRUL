import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  parseMetricsCsv,
  parsePredictionsCsv,
  parseModelManifest,
} from '../src/lib/data/parser';
import {
  getMetricsForDataset,
  getPredictionsForDataset,
  getManifestForDataset,
  calculateCoverage,
  filterAndSortPredictions,
  getSharedParityDomain,
} from '../src/lib/data/adapter';
import {
  TestMetricRecord,
  PredictionRecord,
  ModelManifest,
  VALID_DATASETS,
} from '../src/lib/data/types';
import { LoadedBenchmarkData } from '../src/lib/data/loader';

describe('AeroRUL Data Integration & Contract Verification', () => {
  let metricsCsvRaw: string;
  let predictionsCsvRaw: string;
  let manifestJsonRaw: any;
  let metrics: TestMetricRecord[];
  let predictions: PredictionRecord[];
  let manifest: ModelManifest;
  let benchmarkData: LoadedBenchmarkData;

  beforeAll(() => {
    const dataDir = path.resolve(__dirname, '../public/data/phase5');
    metricsCsvRaw = fs.readFileSync(path.join(dataDir, 'final_test_metrics.csv'), 'utf-8');
    predictionsCsvRaw = fs.readFileSync(path.join(dataDir, 'final_test_predictions.csv'), 'utf-8');
    manifestJsonRaw = JSON.parse(fs.readFileSync(path.join(dataDir, 'final_model_manifest.json'), 'utf-8'));

    metrics = parseMetricsCsv(metricsCsvRaw);
    predictions = parsePredictionsCsv(predictionsCsvRaw);
    manifest = parseModelManifest(manifestJsonRaw);

    benchmarkData = {
      metrics,
      predictions,
      manifest,
      syncedTimestamp: manifest.frozen_timestamp,
    };
  });

  // 1. Correct parsing of the metrics CSV
  it('1. parses metrics CSV correctly with exact row count and fields', () => {
    expect(metrics).toHaveLength(4);
    const datasets = metrics.map((m) => m.dataset);
    expect(datasets).toEqual(['FD001', 'FD002', 'FD003', 'FD004']);

    for (const m of metrics) {
      expect(m.selected_model).toBe('LightGBM');
      expect(Number.isFinite(m.endpoint_mae)).toBe(true);
      expect(Number.isFinite(m.endpoint_rmse)).toBe(true);
      expect(Number.isFinite(m.endpoint_r2)).toBe(true);
      expect(Number.isFinite(m.mean_signed_error)).toBe(true);
      expect(m.n_official_test_engines).toBeGreaterThan(0);
      expect(m.n_successfully_predicted).toBe(m.n_official_test_engines);
    }
  });

  // 2. Correct parsing of prediction records
  it('2. parses prediction records into finite numeric values', () => {
    expect(predictions).toHaveLength(707);
    for (const p of predictions) {
      expect(VALID_DATASETS).toContain(p.dataset);
      expect(p.unit_id).toBeGreaterThan(0);
      expect(p.last_observed_cycle).toBeGreaterThan(0);
      expect(Number.isFinite(p.actual_rul_official)).toBe(true);
      expect(Number.isFinite(p.predicted_rul_final)).toBe(true);
      expect(Number.isFinite(p.prediction_error)).toBe(true);
      expect(Number.isFinite(p.absolute_error)).toBe(true);
    }
  });

  // 3. Correct manifest lookup by dataset
  it('3. looks up manifest configurations correctly for all datasets', () => {
    expect(manifest.manifest_version).toBe('1.0');
    expect(manifest.frozen_timestamp).toBe('2026-10-27 00:35:50');

    for (const ds of VALID_DATASETS) {
      const config = getManifestForDataset(benchmarkData, ds);
      expect(config).toBeDefined();
      expect(config?.dataset).toBe(ds);
      expect(config?.model_family).toBe('LightGBM');
      expect(config?.n_features).toBeGreaterThan(0);
      expect(config?.hyperparameters).toBeDefined();
    }

    // Verify exact feature counts matching frozen reference
    expect(getManifestForDataset(benchmarkData, 'FD001')?.n_features).toBe(670);
    expect(getManifestForDataset(benchmarkData, 'FD002')?.n_features).toBe(231);
    expect(getManifestForDataset(benchmarkData, 'FD003')?.n_features).toBe(681);
    expect(getManifestForDataset(benchmarkData, 'FD004')?.n_features).toBe(768);
  });

  // 4. Correct prediction error interpretation
  it('4. calculates prediction error consistently (signed vs absolute)', () => {
    for (const p of predictions) {
      // Signed error = predicted - actual
      const expectedSigned = p.predicted_rul_final - p.actual_rul_official;
      expect(Math.abs(p.prediction_error - expectedSigned)).toBeLessThan(1e-4);

      // Absolute error = |signed error|
      expect(Math.abs(p.absolute_error - Math.abs(expectedSigned))).toBeLessThan(1e-4);

      // Error categorization
      if (p.prediction_error > 0) {
        expect(p.predicted_rul_final).toBeGreaterThan(p.actual_rul_official);
      } else if (p.prediction_error < 0) {
        expect(p.predicted_rul_final).toBeLessThan(p.actual_rul_official);
      }
    }
  });

  // 5. Correct dataset filtering
  it('5. filters predictions cleanly by dataset without cross-contamination', () => {
    const fd001 = getPredictionsForDataset(benchmarkData, 'FD001');
    const fd002 = getPredictionsForDataset(benchmarkData, 'FD002');
    const fd003 = getPredictionsForDataset(benchmarkData, 'FD003');
    const fd004 = getPredictionsForDataset(benchmarkData, 'FD004');

    expect(fd001.every((p) => p.dataset === 'FD001')).toBe(true);
    expect(fd002.every((p) => p.dataset === 'FD002')).toBe(true);
    expect(fd003.every((p) => p.dataset === 'FD003')).toBe(true);
    expect(fd004.every((p) => p.dataset === 'FD004')).toBe(true);
  });

  // 6. Correct subset record counts
  it('6. verifies exact subset engine counts matching C-MAPSS official benchmark', () => {
    const fd001 = getPredictionsForDataset(benchmarkData, 'FD001');
    const fd002 = getPredictionsForDataset(benchmarkData, 'FD002');
    const fd003 = getPredictionsForDataset(benchmarkData, 'FD003');
    const fd004 = getPredictionsForDataset(benchmarkData, 'FD004');

    expect(fd001).toHaveLength(100);
    expect(fd002).toHaveLength(259);
    expect(fd003).toHaveLength(100);
    expect(fd004).toHaveLength(248);
    expect(fd001.length + fd002.length + fd003.length + fd004.length).toBe(707);
  });

  // 7. Correct metric values to an appropriate precision
  it('7. matches verified official test metric references', () => {
    const m1 = getMetricsForDataset(benchmarkData, 'FD001')!;
    expect(m1.endpoint_mae).toBeCloseTo(11.4348, 3);
    expect(m1.endpoint_rmse).toBeCloseTo(15.6858, 3);
    expect(m1.endpoint_r2).toBeCloseTo(0.8575, 3);
    expect(m1.selected_target_cap).toBe(150);

    const m2 = getMetricsForDataset(benchmarkData, 'FD002')!;
    expect(m2.endpoint_mae).toBeCloseTo(13.7838, 3);
    expect(m2.endpoint_rmse).toBeCloseTo(20.1271, 3);
    expect(m2.endpoint_r2).toBeCloseTo(0.8599, 3);
    expect(m2.selected_target_cap).toBe(150);

    const m3 = getMetricsForDataset(benchmarkData, 'FD003')!;
    expect(m3.endpoint_mae).toBeCloseTo(14.6807, 3);
    expect(m3.endpoint_rmse).toBeCloseTo(25.0027, 3);
    expect(m3.endpoint_r2).toBeCloseTo(0.6352, 3);
    expect(m3.selected_target_cap).toBe('None');

    const m4 = getMetricsForDataset(benchmarkData, 'FD004')!;
    expect(m4.endpoint_mae).toBeCloseTo(22.8125, 3);
    expect(m4.endpoint_rmse).toBeCloseTo(33.8018, 3);
    expect(m4.endpoint_r2).toBeCloseTo(0.6157, 3);
    expect(m4.selected_target_cap).toBe('None');
  });

  // 8. Correct coverage calculation
  it('8. calculates prediction coverage accurately', () => {
    for (const m of metrics) {
      const cov = calculateCoverage(m);
      expect(cov.coverageRatio).toBe(1.0);
      expect(cov.coveragePercent).toBe('100.0');
      expect(cov.successfullyPredicted).toBe(cov.totalOfficial);
    }
  });

  // 9. Correct sorting and searching behaviour
  it('9. supports search by unit ID and multi-column sorting', () => {
    const fd001 = getPredictionsForDataset(benchmarkData, 'FD001');

    // Search
    const searched = filterAndSortPredictions(fd001, '42', { field: 'unit_id', direction: 'asc' });
    expect(searched.length).toBeGreaterThan(0);
    expect(searched.every((p) => String(p.unit_id).includes('42'))).toBe(true);

    // Sort ascending by MAE / absolute_error
    const sortedAsc = filterAndSortPredictions(fd001, '', { field: 'absolute_error', direction: 'asc' });
    for (let i = 0; i < sortedAsc.length - 1; i++) {
      expect(sortedAsc[i].absolute_error).toBeLessThanOrEqual(sortedAsc[i + 1].absolute_error);
    }

    // Sort descending by actual RUL
    const sortedDesc = filterAndSortPredictions(fd001, '', { field: 'actual_rul_official', direction: 'desc' });
    for (let i = 0; i < sortedDesc.length - 1; i++) {
      expect(sortedDesc[i].actual_rul_official).toBeGreaterThanOrEqual(sortedDesc[i + 1].actual_rul_official);
    }
  });

  // 10. Handling of malformed or missing artifacts
  it('10. handles malformed or missing data with explicit errors', () => {
    expect(() => parseMetricsCsv('')).toThrow('Metrics CSV is empty');
    expect(() => parseMetricsCsv('invalid,header\n1,2')).toThrow();
    expect(() => parsePredictionsCsv('')).toThrow('Predictions CSV is empty');
    expect(() => parseModelManifest(null)).toThrow('Manifest JSON is invalid');
    expect(() => parseModelManifest({})).toThrow('Manifest missing top-level keys');
  });

  // Parity domain sanity
  it('computes positive shared parity domain for scatter plots', () => {
    const fd001 = getPredictionsForDataset(benchmarkData, 'FD001');
    const domain = getSharedParityDomain(fd001);
    expect(domain[0]).toBe(0);
    expect(domain[1]).toBeGreaterThanOrEqual(140);
  });
});
