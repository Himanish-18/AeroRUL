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

  describe('Task 2 — Official Summary Statistics Reconciliation Across All 4 Subsets', () => {
    const expectedDatasetStats = [
      {
        dataset: 'FD001' as const,
        total: 100,
        over: 56,
        under: 44,
        exact: 0,
        median: 7.98,
        p90: 27.00,
        mae: 11.43,
      },
      {
        dataset: 'FD002' as const,
        total: 259,
        over: 124,
        under: 135,
        exact: 0,
        median: 8.01,
        p90: 35.77,
        mae: 13.78,
      },
      {
        dataset: 'FD003' as const,
        total: 100,
        over: 73,
        under: 27,
        exact: 0,
        median: 6.79,
        p90: 36.22,
        mae: 14.68,
      },
      {
        dataset: 'FD004' as const,
        total: 248,
        over: 153,
        under: 95,
        exact: 0,
        median: 13.20,
        p90: 57.85,
        mae: 22.81,
      },
    ];

    for (const ref of expectedDatasetStats) {
      it(`verifies exact summary statistics and dispersion for ${ref.dataset}`, () => {
        const preds = getPredictionsForDataset(benchmarkData, ref.dataset);
        const metric = getMetricsForDataset(benchmarkData, ref.dataset)!;
        const summary = getPredictionSummary(preds, metric);

        expect(preds).toHaveLength(ref.total);
        expect(summary.totalEvaluated).toBe(ref.total);
        expect(summary.overpredictionCount).toBe(ref.over);
        expect(summary.underpredictionCount).toBe(ref.under);
        expect(summary.exactCount).toBe(ref.exact);

        // Sum of categories equals total engines
        expect(summary.overpredictionCount + summary.underpredictionCount + summary.exactCount).toBe(ref.total);

        // Consistency with official metric artifact
        expect(summary.overpredictionCount).toBe(metric.overprediction_count);
        expect(summary.underpredictionCount).toBe(metric.underprediction_count);
        expect(summary.endpointMae).toBeCloseTo(metric.endpoint_mae, 4);
        expect(summary.endpointMae).toBeCloseTo(ref.mae, 2);

        // Dispersion metrics
        expect(summary.medianAbsoluteError).toBeCloseTo(ref.median, 2);
        expect(summary.p90AbsoluteError).toBeCloseTo(ref.p90, 2);
      });
    }

    it('handles empty predictions safely', () => {
      const emptySummary = getPredictionSummary([]);
      expect(emptySummary.totalEvaluated).toBe(0);
      expect(emptySummary.coveragePercent).toBe('0.0');
      expect(emptySummary.endpointMae).toBe(0);
      expect(emptySummary.medianAbsoluteError).toBe(0);
      expect(emptySummary.p90AbsoluteError).toBe(0);
    });
  });

  describe('Task 7 — Error Distribution Histogram & Zero Handling', () => {
    it('bins signed errors accurately on real FD001 dataset with total count conserved', () => {
      const bins = getErrorDistribution(fd001Predictions, 10);
      expect(bins.length).toBeGreaterThan(0);

      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(fd001Predictions.length);

      for (const bin of bins) {
        if (bin.binEnd <= 0) {
          expect(bin.isUnderprediction).toBe(true);
          expect(bin.isOverprediction).toBe(false);
        } else if (bin.binStart >= 0) {
          expect(bin.isOverprediction).toBe(true);
          expect(bin.isUnderprediction).toBe(false);
        }
      }
    });

    it('handles negative-only prediction errors correctly', () => {
      const mockNegative: PredictionRecord[] = [
        { ...fd001Predictions[0], prediction_error: -25, absolute_error: 25 },
        { ...fd001Predictions[0], prediction_error: -15, absolute_error: 15 },
        { ...fd001Predictions[0], prediction_error: -5, absolute_error: 5 },
      ];

      const bins = getErrorDistribution(mockNegative, 10);
      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(3);

      expect(bins.every((b) => b.isUnderprediction)).toBe(true);
      expect(bins.some((b) => b.isOverprediction)).toBe(false);
      expect(bins.some((b) => b.isZeroBin)).toBe(false);
    });

    it('handles positive-only prediction errors correctly', () => {
      const mockPositive: PredictionRecord[] = [
        { ...fd001Predictions[0], prediction_error: 5, absolute_error: 5 },
        { ...fd001Predictions[0], prediction_error: 15, absolute_error: 15 },
        { ...fd001Predictions[0], prediction_error: 25, absolute_error: 25 },
      ];

      const bins = getErrorDistribution(mockPositive, 10);
      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(3);

      expect(bins.every((b) => b.isOverprediction)).toBe(true);
      expect(bins.some((b) => b.isUnderprediction)).toBe(false);
      expect(bins.some((b) => b.isZeroBin)).toBe(false);
    });

    it('handles exact zero error as a neutral category without misclassifying as overprediction', () => {
      const mockWithZero: PredictionRecord[] = [
        { ...fd001Predictions[0], prediction_error: -10, absolute_error: 10 },
        { ...fd001Predictions[0], prediction_error: 0, absolute_error: 0 },
        { ...fd001Predictions[0], prediction_error: 10, absolute_error: 10 },
      ];

      const bins = getErrorDistribution(mockWithZero, 10);
      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(3);

      const zeroBin = bins.find((b) => b.isZeroBin);
      expect(zeroBin).toBeDefined();
      expect(zeroBin?.count).toBe(1);
      expect(zeroBin?.isOverprediction).toBe(false);
      expect(zeroBin?.isUnderprediction).toBe(false);
    });

    it('handles errors spanning zero with clear mathematical boundary', () => {
      const mockSpanning: PredictionRecord[] = [
        { ...fd001Predictions[0], prediction_error: -18, absolute_error: 18 },
        { ...fd001Predictions[0], prediction_error: -2, absolute_error: 2 },
        { ...fd001Predictions[0], prediction_error: 4, absolute_error: 4 },
        { ...fd001Predictions[0], prediction_error: 22, absolute_error: 22 },
      ];

      const bins = getErrorDistribution(mockSpanning, 10);
      const totalBinned = bins.reduce((sum, b) => sum + b.count, 0);
      expect(totalBinned).toBe(4);

      // Verify negative bins are strictly <= 0 and positive bins are strictly >= 0
      for (const b of bins) {
        if (b.isUnderprediction) {
          expect(b.binEnd).toBeLessThanOrEqual(0);
        }
        if (b.isOverprediction) {
          expect(b.binStart).toBeGreaterThanOrEqual(0);
        }
      }
    });

    it('returns empty array when given no predictions', () => {
      expect(getErrorDistribution([])).toEqual([]);
    });
  });

  describe('Outlier Identification (getTopOutliers)', () => {
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

  describe('Advanced Filtering & Sorting (filterAndSortPredictionsAdvanced)', () => {
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

  describe('Post-Processing Constraint Audit', () => {
    it('confirms target cap constraint affects raw predictions where cap = 150 (FD001 and FD002)', () => {
      const cappedEngines = fd001Predictions.filter((p) => p.predicted_rul_raw > 150);
      if (cappedEngines.length > 0) {
        for (const p of cappedEngines) {
          expect(p.predicted_rul_final).toBe(150);
          expect(p.predicted_rul_final - p.predicted_rul_raw).toBeLessThan(0);
        }
      }

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
