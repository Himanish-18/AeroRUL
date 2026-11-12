import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { parsePredictionsCsv, parseMetricsCsv, parseModelManifest } from '../src/lib/data/parser';
import {
  getPredictionsForDataset,
  getMetricsForDataset,
  getPredictionSummary,
  getErrorDistribution,
  getTopOutliers,
  filterAndSortPredictionsAdvanced,
} from '../src/lib/data/adapter';
import {
  PredictionRecord,
  TestMetricRecord,
  PredictionFilterState,
  SortState,
} from '../src/lib/data/types';
import { LoadedBenchmarkData } from '../src/lib/data/loader';

describe('AeroRUL Stage 2: Engine Predictions Analysis & Metrics', () => {
  let benchmarkData: LoadedBenchmarkData;
  let fd001Predictions: PredictionRecord[];
  let fd001Metric: TestMetricRecord;

  beforeAll(() => {
    const dataDir = path.resolve(__dirname, '../public/data/phase5');
    const metricsCsvRaw = fs.readFileSync(path.join(dataDir, 'final_test_metrics.csv'), 'utf-8');
    const predictionsCsvRaw = fs.readFileSync(path.join(dataDir, 'final_test_predictions.csv'), 'utf-8');
    const manifestJsonRaw = JSON.parse(fs.readFileSync(path.join(dataDir, 'final_model_manifest.json'), 'utf-8'));

    const metrics = parseMetricsCsv(metricsCsvRaw);
    const predictions = parsePredictionsCsv(predictionsCsvRaw);
    const manifest = parseModelManifest(manifestJsonRaw);

    benchmarkData = {
      metrics,
      predictions,
      manifest,
      syncedTimestamp: manifest.frozen_timestamp,
    };

    fd001Predictions = getPredictionsForDataset(benchmarkData, 'FD001');
    fd001Metric = getMetricsForDataset(benchmarkData, 'FD001')!;
  });

  describe('1. Summary Statistics Calculation (getPredictionSummary)', () => {
    it('accurately computes total evaluated, over/under counts, and percentages for FD001', () => {
      const summary = getPredictionSummary(fd001Predictions, fd001Metric);

      expect(summary.totalEvaluated).toBe(100);
      expect(summary.successfullyPredicted).toBe(100);
      expect(summary.coveragePercent).toBe('100.0');

      // Check sum of over, under, and exact equals total evaluated
      expect(summary.overpredictionCount + summary.underpredictionCount + summary.exactCount).toBe(100);
      expect(summary.overpredictionPercent + summary.underpredictionPercent + summary.exactPercent).toBeCloseTo(100, 1);

      // Verify counts match official metrics record
      expect(summary.overpredictionCount).toBe(fd001Metric.overprediction_count);
      expect(summary.underpredictionCount).toBe(fd001Metric.underprediction_count);

      // MAE matches
      expect(summary.endpointMae).toBeCloseTo(fd001Metric.endpoint_mae, 4);

      // Mean signed error matches
      expect(summary.meanSignedError).toBeCloseTo(fd001Metric.mean_signed_error, 4);
    });

    it('computes valid median and 90th percentile absolute errors', () => {
      const summary = getPredictionSummary(fd001Predictions, fd001Metric);

      expect(summary.medianAbsoluteError).toBeGreaterThan(0);
      expect(summary.p90AbsoluteError).toBeGreaterThan(summary.medianAbsoluteError);
      expect(summary.p90AbsoluteError).toBeGreaterThan(summary.endpointMae);

      // Verify median manually on sorted array
      const sortedAbs = fd001Predictions.map((p) => p.absolute_error).sort((a, b) => a - b);
      const expectedMedian = (sortedAbs[49] + sortedAbs[50]) / 2;
      expect(summary.medianAbsoluteError).toBeCloseTo(expectedMedian, 4);
    });

    it('handles empty predictions safely', () => {
      const emptySummary = getPredictionSummary([]);
      expect(emptySummary.totalEvaluated).toBe(0);
      expect(emptySummary.coveragePercent).toBe('0.0');
      expect(emptySummary.endpointMae).toBe(0);
      expect(emptySummary.medianAbsoluteError).toBe(0);
      expect(emptySummary.p90AbsoluteError).toBe(0);
    });
  });

  describe('2. Error Distribution Histogram (getErrorDistribution)', () => {
    it('bins signed errors accurately with continuous cycle ranges', () => {
      const bins = getErrorDistribution(fd001Predictions, 10);
      expect(bins.length).toBeGreaterThan(0);

      // Sum of counts across all bins must equal total predictions
      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(fd001Predictions.length);

      // Check that bins are contiguous
      for (let i = 0; i < bins.length - 1; i++) {
        expect(bins[i].binEnd).toBe(bins[i + 1].binStart);
      }

      // Check classification properties
      for (const bin of bins) {
        if (bin.binEnd <= 0) {
          expect(bin.isUnderprediction).toBe(true);
        } else if (bin.binStart >= 0) {
          expect(bin.isOverprediction).toBe(true);
        }
      }
    });

    it('returns empty array when given no predictions', () => {
      expect(getErrorDistribution([])).toEqual([]);
    });
  });

  describe('3. Outlier Identification (getTopOutliers)', () => {
    it('returns exactly top N outliers in descending order of absolute error', () => {
      const top5 = getTopOutliers(fd001Predictions, 5);
      expect(top5).toHaveLength(5);

      for (let i = 0; i < top5.length - 1; i++) {
        expect(top5[i].absolute_error).toBeGreaterThanOrEqual(top5[i + 1].absolute_error);
      }

      // Ensure every other prediction in the dataset has absolute error <= top5[4]
      const minTop5Error = top5[4].absolute_error;
      const top5Ids = new Set(top5.map((p) => p.unit_id));
      for (const p of fd001Predictions) {
        if (!top5Ids.has(p.unit_id)) {
          expect(p.absolute_error).toBeLessThanOrEqual(minTop5Error);
        }
      }
    });
  });

  describe('4. Advanced Filtering & Sorting (filterAndSortPredictionsAdvanced)', () => {
    const baseSort: SortState = { field: 'unit_id', direction: 'asc' };

    it('filters correctly by engine unit_id substring', () => {
      const filters: PredictionFilterState = {
        searchQuery: '10',
        errorClassification: 'all',
        minAbsoluteError: 0,
      };
      const result = filterAndSortPredictionsAdvanced(fd001Predictions, filters, baseSort);
      expect(result.length).toBeGreaterThan(0);
      expect(result.every((p) => String(p.unit_id).includes('10'))).toBe(true);
    });

    it('filters correctly by overprediction classification', () => {
      const filters: PredictionFilterState = {
        searchQuery: '',
        errorClassification: 'over',
        minAbsoluteError: 0,
      };
      const result = filterAndSortPredictionsAdvanced(fd001Predictions, filters, baseSort);
      expect(result.length).toBe(fd001Metric.overprediction_count);
      expect(result.every((p) => p.prediction_error > 0)).toBe(true);
    });

    it('filters correctly by underprediction classification', () => {
      const filters: PredictionFilterState = {
        searchQuery: '',
        errorClassification: 'under',
        minAbsoluteError: 0,
      };
      const result = filterAndSortPredictionsAdvanced(fd001Predictions, filters, baseSort);
      expect(result.length).toBe(fd001Metric.underprediction_count);
      expect(result.every((p) => p.prediction_error < 0)).toBe(true);
    });

    it('filters correctly by minimum absolute error threshold', () => {
      const threshold = 20;
      const filters: PredictionFilterState = {
        searchQuery: '',
        errorClassification: 'all',
        minAbsoluteError: threshold,
      };
      const result = filterAndSortPredictionsAdvanced(fd001Predictions, filters, baseSort);
      expect(result.length).toBeGreaterThan(0);
      expect(result.every((p) => p.absolute_error >= threshold)).toBe(true);
    });

    it('combines search, classification, and threshold filters simultaneously', () => {
      const filters: PredictionFilterState = {
        searchQuery: '1',
        errorClassification: 'under',
        minAbsoluteError: 15,
      };
      const result = filterAndSortPredictionsAdvanced(fd001Predictions, filters, baseSort);
      for (const p of result) {
        expect(String(p.unit_id)).toContain('1');
        expect(p.prediction_error).toBeLessThan(0);
        expect(p.absolute_error).toBeGreaterThanOrEqual(15);
      }
    });

    it('sorts multi-column fields ascending and descending properly', () => {
      const allFilters: PredictionFilterState = {
        searchQuery: '',
        errorClassification: 'all',
        minAbsoluteError: 0,
      };

      // Raw predicted RUL ascending
      const sortedRawAsc = filterAndSortPredictionsAdvanced(
        fd001Predictions,
        allFilters,
        { field: 'predicted_rul_raw', direction: 'asc' }
      );
      for (let i = 0; i < sortedRawAsc.length - 1; i++) {
        expect(sortedRawAsc[i].predicted_rul_raw).toBeLessThanOrEqual(sortedRawAsc[i + 1].predicted_rul_raw);
      }

      // Signed error descending
      const sortedErrorDesc = filterAndSortPredictionsAdvanced(
        fd001Predictions,
        allFilters,
        { field: 'prediction_error', direction: 'desc' }
      );
      for (let i = 0; i < sortedErrorDesc.length - 1; i++) {
        expect(sortedErrorDesc[i].prediction_error).toBeGreaterThanOrEqual(sortedErrorDesc[i + 1].prediction_error);
      }
    });
  });

  describe('5. Post-Processing Constraint Audit', () => {
    it('confirms target cap constraint affects raw predictions where cap = 150 (FD001 and FD002)', () => {
      // FD001 has target cap 150
      const cappedEngines = fd001Predictions.filter((p) => p.predicted_rul_raw > 150);
      if (cappedEngines.length > 0) {
        for (const p of cappedEngines) {
          expect(p.predicted_rul_final).toBe(150);
          expect(p.predicted_rul_final - p.predicted_rul_raw).toBeLessThan(0);
        }
      }

      // All FD001 final predictions should be <= 150
      for (const p of fd001Predictions) {
        expect(p.predicted_rul_final).toBeLessThanOrEqual(150);
      }
    });

    it('confirms unconstrained models (FD003 & FD004) preserve raw outputs above 150 cycles', () => {
      const fd003Predictions = getPredictionsForDataset(benchmarkData, 'FD003');
      const unconstrainedAbove150 = fd003Predictions.filter((p) => p.predicted_rul_raw > 150);
      if (unconstrainedAbove150.length > 0) {
        for (const p of unconstrainedAbove150) {
          expect(p.predicted_rul_final).toBe(p.predicted_rul_raw);
        }
      }
    });
  });
});
