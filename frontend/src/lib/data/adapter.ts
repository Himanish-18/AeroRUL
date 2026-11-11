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
