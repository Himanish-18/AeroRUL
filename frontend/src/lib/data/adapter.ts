/**
 * AeroRUL — Data Adapter
 *
 * Provides query selectors, dataset filtering, sorting, coverage calculation,
 * and scatter coordinate scaling without mutating source arrays.
 */

import {
  DatasetId,
  TestMetricRecord,
  PredictionRecord,
  DatasetManifestConfig,
  SortState,
  PredictionSummaryStats,
  ErrorDistributionBin,
  PredictionFilterState,
} from './types';
import { LoadedBenchmarkData } from './loader';

export function getMetricsForDataset(
  data: LoadedBenchmarkData,
  dataset: DatasetId
): TestMetricRecord | undefined {
  return data.metrics.find((m) => m.dataset === dataset);
}

export function getPredictionsForDataset(
  data: LoadedBenchmarkData,
  dataset: DatasetId
): PredictionRecord[] {
  return data.predictions.filter((p) => p.dataset === dataset);
}

export function getManifestForDataset(
  data: LoadedBenchmarkData,
  dataset: DatasetId
): DatasetManifestConfig | undefined {
  return data.manifest.selected_configurations[dataset];
}

export function calculateCoverage(metric: TestMetricRecord): {
  coverageRatio: number;
  coveragePercent: string;
  successfullyPredicted: number;
  totalOfficial: number;
} {
  const successfullyPredicted = metric.n_successfully_predicted;
  const totalOfficial = metric.n_official_test_engines;
  const ratio = totalOfficial > 0 ? successfullyPredicted / totalOfficial : 0;
  return {
    coverageRatio: ratio,
    coveragePercent: (ratio * 100).toFixed(1),
    successfullyPredicted,
    totalOfficial,
  };
}

export function filterAndSortPredictions(
  predictions: PredictionRecord[],
  searchQuery: string,
  sort: SortState
): PredictionRecord[] {
  let filtered = predictions;

  const trimmed = searchQuery.trim();
  if (trimmed) {
    const qNum = parseInt(trimmed, 10);
    if (!Number.isNaN(qNum)) {
      filtered = filtered.filter((p) => String(p.unit_id).includes(trimmed));
    }
  }

  const sorted = [...filtered].sort((a, b) => {
    const valA = a[sort.field];
    const valB = b[sort.field];

    if (typeof valA === 'number' && typeof valB === 'number') {
      return sort.direction === 'asc' ? valA - valB : valB - valA;
    }

    const strA = String(valA);
    const strB = String(valB);
    return sort.direction === 'asc'
      ? strA.localeCompare(strB)
      : strB.localeCompare(strA);
  });

  return sorted;
}

/**
 * Calculates a shared symmetrical domain [0, max] for both axes of the scatter plot
 * to ensure that the y = x parity reference line aligns at 45 degrees.
 */
export function getSharedParityDomain(predictions: PredictionRecord[]): [number, number] {
  if (predictions.length === 0) return [0, 150];

  let maxVal = 0;
  for (const p of predictions) {
    if (p.actual_rul_official > maxVal) maxVal = p.actual_rul_official;
    if (p.predicted_rul_final > maxVal) maxVal = p.predicted_rul_final;
  }

  // Round up to nearest comfortable tick interval (e.g. 25 or 50)
  const ceiling = Math.ceil(maxVal / 25) * 25;
  return [0, Math.max(ceiling, 50)];
}

/**
 * Computes summary statistics including over/under-prediction counts and percentages,
 * official MAE, and derived dispersion metrics (median, 90th percentile absolute error).
 */
export function getPredictionSummary(
  predictions: PredictionRecord[],
  metric?: TestMetricRecord
): PredictionSummaryStats {
  const n = predictions.length;
  if (n === 0) {
    return {
      totalEvaluated: 0,
      successfullyPredicted: 0,
      coveragePercent: '0.0',
      overpredictionCount: 0,
      overpredictionPercent: 0,
      underpredictionCount: 0,
      underpredictionPercent: 0,
      exactCount: 0,
      exactPercent: 0,
      endpointMae: 0,
      medianAbsoluteError: 0,
      p90AbsoluteError: 0,
      meanSignedError: 0,
      nasaScoreSum: 0,
    };
  }

  let overCount = 0;
  let underCount = 0;
  let exactCount = 0;
  let absErrorSum = 0;
  let signedErrorSum = 0;

  for (const p of predictions) {
    if (p.prediction_error > 0) {
      overCount++;
    } else if (p.prediction_error < 0) {
      underCount++;
    } else {
      exactCount++;
    }
    absErrorSum += p.absolute_error;
    signedErrorSum += p.prediction_error;
  }

  // Sorted absolute errors for median and 90th percentile
  const sortedAbsErrors = predictions
    .map((p) => p.absolute_error)
    .sort((a, b) => a - b);

  let medianAbsError: number;
  if (n % 2 === 1) {
    medianAbsError = sortedAbsErrors[(n - 1) / 2];
  } else {
    medianAbsError = (sortedAbsErrors[n / 2 - 1] + sortedAbsErrors[n / 2]) / 2;
  }

  // 90th percentile using standard linear interpolation
  const p90Index = 0.9 * (n - 1);
  const p90Lower = Math.floor(p90Index);
  const p90Upper = Math.ceil(p90Index);
  const p90Weight = p90Index - p90Lower;
  const p90AbsError =
    p90Lower === p90Upper
      ? sortedAbsErrors[p90Lower]
      : sortedAbsErrors[p90Lower] * (1 - p90Weight) + sortedAbsErrors[p90Upper] * p90Weight;

  const endpointMae = metric ? metric.endpoint_mae : absErrorSum / n;
  const meanSignedError = metric ? metric.mean_signed_error : signedErrorSum / n;
  const totalOfficial = metric ? metric.n_official_test_engines : n;
  const successfullyPredicted = metric ? metric.n_successfully_predicted : n;
  const coverageRatio = totalOfficial > 0 ? successfullyPredicted / totalOfficial : 1;

  return {
    totalEvaluated: n,
    successfullyPredicted,
    coveragePercent: (coverageRatio * 100).toFixed(1),
    overpredictionCount: overCount,
    overpredictionPercent: (overCount / n) * 100,
    underpredictionCount: underCount,
    underpredictionPercent: (underCount / n) * 100,
    exactCount,
    exactPercent: (exactCount / n) * 100,
    endpointMae,
    medianAbsoluteError: medianAbsError,
    p90AbsoluteError: p90AbsError,
    meanSignedError,
    nasaScoreSum: metric ? metric.nasa_score_sum : 0,
  };
}

/**
 * Computes frequency distribution bins of signed errors around 0.
 * Default bin width is 10 cycles.
 *
 * Guarantees:
 * - Negative signed errors represent underpredictions (< 0).
 * - Positive signed errors represent overpredictions (> 0).
 * - Exact zero errors are never counted as overpredictions.
 * - Negative and positive bins never straddle 0.
 * - Total binned count strictly equals input predictions length.
 */
export function getErrorDistribution(
  predictions: PredictionRecord[],
  binWidth = 10
): ErrorDistributionBin[] {
  if (predictions.length === 0) return [];

  const underList = predictions.filter((p) => p.prediction_error < 0);
  const exactList = predictions.filter((p) => p.prediction_error === 0);
  const overList = predictions.filter((p) => p.prediction_error > 0);

  const bins: ErrorDistributionBin[] = [];

  // 1. Negative bins (Underpredictions: conservative RUL estimate)
  if (underList.length > 0) {
    let minNegative = 0;
    for (const p of underList) {
      if (p.prediction_error < minNegative) minNegative = p.prediction_error;
    }
    const start = Math.floor(minNegative / binWidth) * binWidth;

    for (let b = start; b < 0; b += binWidth) {
      const binEnd = b + binWidth;
      // Match errors in [b, binEnd)
      const count = underList.filter(
        (p) => p.prediction_error >= b && p.prediction_error < binEnd
      ).length;

      bins.push({
        binStart: b,
        binEnd,
        binLabel: `${b} to ${binEnd}`,
        count,
        percentage: (count / predictions.length) * 100,
        isUnderprediction: true,
        isOverprediction: false,
        isZeroBin: false,
      });
    }
  }

  // 2. Exact Zero matches (Neutral / Exact parity)
  if (exactList.length > 0) {
    bins.push({
      binStart: 0,
      binEnd: 0,
      binLabel: '0 (Exact)',
      count: exactList.length,
      percentage: (exactList.length / predictions.length) * 100,
      isUnderprediction: false,
      isOverprediction: false,
      isZeroBin: true,
    });
  }

  // 3. Positive bins (Overpredictions: optimistic RUL estimate)
  if (overList.length > 0) {
    let maxPositive = 0;
    for (const p of overList) {
      if (p.prediction_error > maxPositive) maxPositive = p.prediction_error;
    }
    const end = Math.ceil(maxPositive / binWidth) * binWidth;

    for (let b = 0; b < end; b += binWidth) {
      const binEnd = b + binWidth;
      // Match errors in (b, binEnd]
      const count = overList.filter(
        (p) => p.prediction_error > b && p.prediction_error <= binEnd
      ).length;

      bins.push({
        binStart: b,
        binEnd,
        binLabel: `+${b} to +${binEnd}`,
        count,
        percentage: (count / predictions.length) * 100,
        isUnderprediction: false,
        isOverprediction: true,
        isZeroBin: false,
      });
    }
  }

  return bins;
}

/**
 * Returns the top N engines with largest absolute prediction error.
 */
export function getTopOutliers(
  predictions: PredictionRecord[],
  count = 5
): PredictionRecord[] {
  return [...predictions]
    .sort((a, b) => b.absolute_error - a.absolute_error)
    .slice(0, count);
}

/**
 * Advanced filtering and sorting for the Engine Predictions workspace.
 */
export function filterAndSortPredictionsAdvanced(
  predictions: PredictionRecord[],
  filters: PredictionFilterState,
  sort: SortState
): PredictionRecord[] {
  let result = predictions;

  // Search query filter (unit_id)
  const query = filters.searchQuery.trim();
  if (query) {
    result = result.filter((p) => String(p.unit_id).includes(query));
  }

  // Error classification filter
  if (filters.errorClassification === 'over') {
    result = result.filter((p) => p.prediction_error > 0);
  } else if (filters.errorClassification === 'under') {
    result = result.filter((p) => p.prediction_error < 0);
  }

  // Minimum absolute error filter
  if (filters.minAbsoluteError > 0) {
    result = result.filter((p) => p.absolute_error >= filters.minAbsoluteError);
  }

  // Sorting
  const sorted = [...result].sort((a, b) => {
    const valA = a[sort.field];
    const valB = b[sort.field];

    if (typeof valA === 'number' && typeof valB === 'number') {
      return sort.direction === 'asc' ? valA - valB : valB - valA;
    }

    const strA = String(valA);
    const strB = String(valB);
    return sort.direction === 'asc'
      ? strA.localeCompare(strB)
      : strB.localeCompare(strA);
  });

  return sorted;
}

export * from './modelEvaluationAdapter';

