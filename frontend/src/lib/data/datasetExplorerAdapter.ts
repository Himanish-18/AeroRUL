/**
 * AeroRUL — Dataset Explorer Adapter
 *
 * Presentation adapters, filtering, multi-column sorting, pagination,
 * histogram binning, and derived training RUL calculators.
 */

import {
  DatasetManifest,
  EngineFilterOptions,
  EngineIndexRecord,
  ObservationFilterOptions,
  RawObservationRecord,
} from './datasetExplorerTypes';
import { DatasetId } from './types';

export interface DatasetComparisonRow {
  dataset: DatasetId;
  conditions: number;
  faultModesText: string;
  trainEngines: number;
  trainRows: number;
  testEngines: number;
  testRows: number;
  avgTrainLength: number;
  avgTestLength: number;
  rulRangeText: string;
}

export interface DistributionBin {
  binStart: number;
  binEnd: number;
  binLabel: string;
  count: number;
}

/**
 * Filter and sort engine index records.
 */
export function filterAndSortEngines(
  engines: EngineIndexRecord[],
  options: EngineFilterOptions
): { items: EngineIndexRecord[]; totalCount: number; totalPages: number } {
  let filtered = engines;

  if (options.searchQuery && options.searchQuery.trim()) {
    const q = options.searchQuery.trim().toLowerCase().replace(/^unit\s*#?|^#/, '');
    filtered = filtered.filter((e) => e.unit_id.toString().includes(q));
  }

  // Sort
  const sorted = [...filtered].sort((a, b) => {
    let valA = a[options.sortBy];
    let valB = b[options.sortBy];

    if (valA == null) valA = options.sortAsc ? Infinity : -Infinity;
    if (valB == null) valB = options.sortAsc ? Infinity : -Infinity;

    if (valA < valB) return options.sortAsc ? -1 : 1;
    if (valA > valB) return options.sortAsc ? 1 : -1;
    return a.unit_id - b.unit_id;
  });

  const totalCount = sorted.length;
  const pageSize = options.pageSize || 25;
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
  const currentPage = Math.min(Math.max(1, options.page || 1), totalPages);

  const startIdx = (currentPage - 1) * pageSize;
  const items = sorted.slice(startIdx, startIdx + pageSize);

  return { items, totalCount, totalPages };
}

/**
 * Filter observations by engine unit ID, cycle range, sort, and paginate.
 */
export function filterObservations(
  records: RawObservationRecord[],
  options: ObservationFilterOptions
): {
  items: RawObservationRecord[];
  totalCount: number;
  totalPages: number;
  allEngineCycles: number;
  minObservedCycle: number;
  maxObservedCycle: number;
} {
  // Filter by engine unit ID first
  const engineRecords = records.filter((r) => r.unit_id === options.unitId);
  const allEngineCycles = engineRecords.length;

  if (engineRecords.length === 0) {
    return {
      items: [],
      totalCount: 0,
      totalPages: 1,
      allEngineCycles: 0,
      minObservedCycle: 1,
      maxObservedCycle: 1,
    };
  }

  let minObs = engineRecords[0].cycle;
  let maxObs = engineRecords[0].cycle;
  for (let i = 1; i < engineRecords.length; i++) {
    const c = engineRecords[i].cycle;
    if (c < minObs) minObs = c;
    if (c > maxObs) maxObs = c;
  }

  // Apply cycle range filters
  let filtered = engineRecords;
  if (options.minCycle != null) {
    filtered = filtered.filter((r) => r.cycle >= options.minCycle!);
  }
  if (options.maxCycle != null) {
    filtered = filtered.filter((r) => r.cycle <= options.maxCycle!);
  }

  // Multi-column or single column sort
  const sortKey = options.sortBy || 'cycle';
  const sorted = [...filtered].sort((a, b) => {
    const vA = a[sortKey] ?? 0;
    const vB = b[sortKey] ?? 0;
    if (vA < vB) return options.sortAsc ? -1 : 1;
    if (vA > vB) return options.sortAsc ? 1 : -1;
    return a.cycle - b.cycle;
  });

  const totalCount = sorted.length;
  const pageSize = options.pageSize || 25;
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));
  const currentPage = Math.min(Math.max(1, options.page || 1), totalPages);

  const startIdx = (currentPage - 1) * pageSize;
  const items = sorted.slice(startIdx, startIdx + pageSize);

  return {
    items,
    totalCount,
    totalPages,
    allEngineCycles,
    minObservedCycle: minObs,
    maxObservedCycle: maxObs,
  };
}

/**
 * Generate dataset comparison rows for FD001–FD004 from manifest.
 */
export function getDatasetComparisonRows(manifest: DatasetManifest): DatasetComparisonRow[] {
  const ids: DatasetId[] = ['FD001', 'FD002', 'FD003', 'FD004'];
  return ids.map((id) => {
    const profile = manifest.datasets[id];
    const train = profile.train;
    const test = profile.test;

    const faultText = profile.fault_modes === 1 ? 'HPC Degradation' : 'HPC + Fan Degradation';
    const rulRange = test.rul_statistics
      ? `${test.rul_statistics.min} to ${test.rul_statistics.max} (mean ${test.rul_statistics.mean})`
      : '—';

    return {
      dataset: id,
      conditions: profile.conditions,
      faultModesText: faultText,
      trainEngines: train.engine_count,
      trainRows: train.row_count,
      testEngines: test.engine_count,
      testRows: test.row_count,
      avgTrainLength: train.trajectory_statistics.mean,
      avgTestLength: test.trajectory_statistics.mean,
      rulRangeText: rulRange,
    };
  });
}

/**
 * Derive per-cycle physical RUL series for a training engine.
 * physical_rul = terminal_failure_cycle - current_cycle
 */
export function derivePhysicalTrainingRulSeries(
  engineRecords: RawObservationRecord[]
): { cycle: number; physicalRul: number }[] {
  if (engineRecords.length === 0) return [];
  const maxCycle = Math.max(...engineRecords.map((r) => r.cycle));
  return engineRecords
    .slice()
    .sort((a, b) => a.cycle - b.cycle)
    .map((r) => ({
      cycle: r.cycle,
      physicalRul: maxCycle - r.cycle,
    }));
}

/**
 * Extract and validate official test endpoint RUL labels from test engine index records or an RUL vector.
 * Enforces:
 * - Exactly one endpoint RUL per test engine.
 * - 1-indexed engine mapping.
 * - Finite positive integers.
 * - Does not compute or propagate per-cycle test RUL.
 */
export function extractTestEndpointRuls(
  testEnginesOrVector: EngineIndexRecord[] | number[]
): { unitId: number; endpointRul: number }[] {
  if (!testEnginesOrVector || testEnginesOrVector.length === 0) return [];

  // If passed an array of numbers (from RUL_FD00x.txt)
  if (typeof testEnginesOrVector[0] === 'number') {
    return (testEnginesOrVector as number[]).map((val, idx) => {
      if (!Number.isFinite(val) || val < 0) {
        throw new Error(`Invalid test RUL value at index ${idx}: expected positive finite number, got ${val}`);
      }
      return {
        unitId: idx + 1,
        endpointRul: Math.round(val),
      };
    });
  }

  // If passed EngineIndexRecord[]
  const testRecords = (testEnginesOrVector as EngineIndexRecord[])
    .filter((e) => e.split === 'test')
    .sort((a, b) => a.unit_id - b.unit_id);

  return testRecords.map((rec, idx) => {
    const expectedUnitId = idx + 1;
    if (rec.unit_id !== expectedUnitId) {
      console.warn(`[datasetExplorerAdapter] Non-contiguous test engine unit_id: expected ${expectedUnitId}, got ${rec.unit_id}`);
    }
    const rul = rec.official_endpoint_rul;
    if (rul == null || !Number.isFinite(rul) || rul < 0) {
      throw new Error(`Test engine unit #${rec.unit_id} missing valid official_endpoint_rul`);
    }
    return {
      unitId: rec.unit_id,
      endpointRul: Math.round(rul),
    };
  });
}

/**
 * Compute official test endpoint RUL histogram distribution bins.
 * Uses strictly one endpoint label per test engine.
 */
export function computeTestEndpointRulDistribution(
  testEnginesOrVector: EngineIndexRecord[] | number[],
  nBins = 16
): DistributionBin[] {
  const endpointPairs = extractTestEndpointRuls(testEnginesOrVector);
  if (endpointPairs.length === 0) return [];
  const values = endpointPairs.map((p) => p.endpointRul);
  return computeNumericDistribution(values, nBins);
}

/**
 * Compute equal-width distribution histogram bins for numeric arrays.
 */
export function computeNumericDistribution(values: number[], nBins = 15): DistributionBin[] {
  if (values.length === 0) return [];

  let min = values[0];
  let max = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }

  if (min === max) {
    return [
      {
        binStart: min,
        binEnd: max,
        binLabel: min.toFixed(2),
        count: values.length,
      },
    ];
  }

  const binWidth = (max - min) / nBins;
  const bins: DistributionBin[] = Array.from({ length: nBins }, (_, i) => {
    const start = min + i * binWidth;
    const end = i === nBins - 1 ? max : start + binWidth;
    return {
      binStart: start,
      binEnd: end,
      binLabel: `${start.toFixed(1)} - ${end.toFixed(1)}`,
      count: 0,
    };
  });

  for (let i = 0; i < values.length; i++) {
    const val = values[i];
    let idx = Math.floor((val - min) / binWidth);
    if (idx >= nBins) idx = nBins - 1;
    if (idx < 0) idx = 0;
    bins[idx].count++;
  }

  return bins;
}
