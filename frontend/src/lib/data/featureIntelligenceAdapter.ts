/**
 * AeroRUL — Feature Intelligence Data Adapter
 *
 * Domain transformations, ranking, family aggregations, correlation quadrant analysis,
 * trajectory chart pivoting, and sensor statistics.
 */

import {
  DatasetId,
  FeatureImportanceRecord,
  SensorImportanceRecord,
  FeatureFamilyRecord,
  SensorTrajectoryRecord,
  SensorTrajectoriesManifest,
  SensorMetadata,
} from './types';

export const STATIC_SENSOR_SPECS: Record<string, SensorMetadata> = {
  sensor_1: { name: 'Total temp at fan inlet', unit: '°R', type: 'Temperature' },
  sensor_2: { name: 'Total temp at LPC outlet', unit: '°R', type: 'Temperature' },
  sensor_3: { name: 'Total temp at HPC outlet', unit: '°R', type: 'Temperature' },
  sensor_4: { name: 'Total temp at LPT outlet', unit: '°R', type: 'Temperature' },
  sensor_5: { name: 'Pressure at fan inlet', unit: 'psia', type: 'Pressure' },
  sensor_6: { name: 'Total pressure in bypass duct', unit: 'psia', type: 'Pressure' },
  sensor_7: { name: 'Total pressure at HPC outlet', unit: 'psia', type: 'Pressure' },
  sensor_8: { name: 'Physical fan speed', unit: 'rpm', type: 'Speed' },
  sensor_9: { name: 'Physical core speed', unit: 'rpm', type: 'Speed' },
  sensor_10: { name: 'Engine pressure ratio (P50/P2)', unit: '—', type: 'Ratio' },
  sensor_11: { name: 'Static pressure at HPC outlet', unit: 'psia', type: 'Pressure' },
  sensor_12: { name: 'Ratio of fuel flow to Ps30', unit: 'pps/psia', type: 'Ratio' },
  sensor_13: { name: 'Corrected fan speed', unit: 'rpm', type: 'Speed' },
  sensor_14: { name: 'Corrected core speed', unit: 'rpm', type: 'Speed' },
  sensor_15: { name: 'Bypass ratio', unit: '—', type: 'Ratio' },
  sensor_16: { name: 'Burner fuel-air ratio', unit: '—', type: 'Ratio' },
  sensor_17: { name: 'Bleed enthalpy', unit: '—', type: 'Enthalpy' },
  sensor_18: { name: 'Demanded fan speed', unit: 'rpm', type: 'Speed' },
  sensor_19: { name: 'Demanded corrected fan speed', unit: 'rpm', type: 'Speed' },
  sensor_20: { name: 'HPT coolant bleed', unit: 'lbm/s', type: 'Flow' },
  sensor_21: { name: 'LPT coolant bleed', unit: 'lbm/s', type: 'Flow' },
};

export function getSensorSpec(
  sensor: string,
  manifest?: SensorTrajectoriesManifest | null
): SensorMetadata {
  if (manifest?.sensor_specifications?.[sensor]) {
    return manifest.sensor_specifications[sensor];
  }
  if (STATIC_SENSOR_SPECS[sensor]) {
    return STATIC_SENSOR_SPECS[sensor];
  }
  return { name: sensor, unit: '—', type: 'Telemetry' };
}

export function getFilteredFeatures(
  records: FeatureImportanceRecord[],
  metric: 'gain' | 'split',
  topN: number | 'all',
  searchTerm = ''
): FeatureImportanceRecord[] {
  let list = records;

  const term = searchTerm.trim().toLowerCase();
  if (term) {
    list = list.filter((r) =>
      r.feature_name.toLowerCase().includes(term) ||
      r.sensor.toLowerCase().includes(term) ||
      r.statistic.toLowerCase().includes(term) ||
      r.description.toLowerCase().includes(term)
    );
  }

  // Sort by requested metric descending
  const sorted = [...list].sort((a, b) => {
    if (metric === 'gain') {
      return b.gain_importance - a.gain_importance;
    }
    return b.split_importance - a.split_importance;
  });

  if (topN === 'all') {
    return sorted;
  }
  return sorted.slice(0, topN);
}

export function getSensorImportanceList(
  records: SensorImportanceRecord[],
  metric: 'gain' | 'split'
): SensorImportanceRecord[] {
  return [...records].sort((a, b) => {
    if (metric === 'gain') {
      return b.gain_importance - a.gain_importance;
    }
    return b.split_importance - a.split_importance;
  });
}

export function getFamilyBreakdownList(
  records: FeatureFamilyRecord[],
  dataset: DatasetId,
  familyType: 'sensor' | 'statistic' | 'window'
): FeatureFamilyRecord[] {
  return records
    .filter((r) => r.dataset === dataset && r.family_type === familyType)
    .sort((a, b) => b.gain_importance - a.gain_importance);
}

export type QuadrantType =
  | 'high_importance_high_corr'
  | 'high_importance_low_corr'
  | 'low_importance_high_corr'
  | 'low_importance_low_corr';

export interface CorrelationComparisonItem extends FeatureImportanceRecord {
  quadrant: QuadrantType;
  quadrantLabel: string;
}

export function getCorrelationComparisonList(
  records: FeatureImportanceRecord[]
): CorrelationComparisonItem[] {
  // Only evaluate features with at least one correlation value
  const candidates = records.filter(
    (r) => r.abs_pearson !== null || r.abs_spearman !== null
  );

  return candidates.map((r) => {
    const corr = r.abs_pearson ?? r.abs_spearman ?? 0;
    const isHighGain = r.gain_importance_pct >= 2.0;
    const isHighCorr = corr >= 0.5;

    let quadrant: QuadrantType;
    let quadrantLabel: string;

    if (isHighGain && isHighCorr) {
      quadrant = 'high_importance_high_corr';
      quadrantLabel = 'Primary Signal (High Gain & High Correlation)';
    } else if (isHighGain && !isHighCorr) {
      quadrant = 'high_importance_low_corr';
      quadrantLabel = 'Nonlinear / Synergistic (High Gain & Moderate Correlation)';
    } else if (!isHighGain && isHighCorr) {
      quadrant = 'low_importance_high_corr';
      quadrantLabel = 'Redundant / Shared Signal (High Correlation & Low Gain)';
    } else {
      quadrant = 'low_importance_low_corr';
      quadrantLabel = 'Subordinate Feature (Low Gain & Low Correlation)';
    }

    return {
      ...r,
      quadrant,
      quadrantLabel,
    };
  });
}

export interface TrajectoryPoint {
  cycle: number;
  [unitKey: string]: number;
}

export function getTrajectoryChartData(
  records: SensorTrajectoryRecord[],
  sensor: string,
  selectedUnits: number[]
): TrajectoryPoint[] {
  if (!records || records.length === 0 || selectedUnits.length === 0) {
    return [];
  }

  const filtered = records.filter(
    (r) => r.sensor === sensor && selectedUnits.includes(r.unit_id)
  );

  const cycleMap = new Map<number, TrajectoryPoint>();

  for (const row of filtered) {
    let point = cycleMap.get(row.cycle);
    if (!point) {
      point = { cycle: row.cycle };
      cycleMap.set(row.cycle, point);
    }
    point[`unit_${row.unit_id}`] = row.sensor_value;
  }

  return Array.from(cycleMap.values()).sort((a, b) => a.cycle - b.cycle);
}

export interface EngineSensorStats {
  unitId: number;
  sensor: string;
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  startCycle: number;
  endCycle: number;
  startValue: number;
  endValue: number;
  netChange: number;
}

export function getEngineSensorStatistics(
  records: SensorTrajectoryRecord[],
  sensor: string,
  unitId: number
): EngineSensorStats | null {
  const engineRows = records
    .filter((r) => r.sensor === sensor && r.unit_id === unitId)
    .sort((a, b) => a.cycle - b.cycle);

  if (engineRows.length === 0) {
    return null;
  }

  const values = engineRows.map((r) => r.sensor_value);
  const count = values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((sum, v) => sum + v, 0) / count;

  const variance =
    values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (count > 1 ? count - 1 : 1);
  const std = Math.sqrt(variance);

  const startRow = engineRows[0];
  const endRow = engineRows[count - 1];

  return {
    unitId,
    sensor,
    count,
    min,
    max,
    mean,
    std,
    startCycle: startRow.cycle,
    endCycle: endRow.cycle,
    startValue: startRow.sensor_value,
    endValue: endRow.sensor_value,
    netChange: endRow.sensor_value - startRow.sensor_value,
  };
}

export const ENGINE_COLOR_PALETTE = [
  '#f59e0b', // Amber (Primary)
  '#38bdf8', // Light Blue / Cyan
  '#10b981', // Emerald
  '#a855f7', // Purple
  '#f43f5e', // Rose
  '#fb923c', // Orange
  '#60a5fa', // Blue
  '#a3e635', // Lime
  '#fde047', // Yellow
  '#e879f9', // Fuchsia
];

export function getEngineColor(index: number): string {
  return ENGINE_COLOR_PALETTE[index % ENGINE_COLOR_PALETTE.length];
}
