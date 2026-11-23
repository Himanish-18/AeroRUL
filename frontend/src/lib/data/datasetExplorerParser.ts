/**
 * AeroRUL — Dataset Explorer Parser
 *
 * Strict, robust parsing for C-MAPSS whitespace-delimited raw tables,
 * engine index CSVs, and RUL ground-truth vectors.
 */

import Papa from 'papaparse';
import {
  EngineIndexRecord,
  QualityStats,
  RawObservationRecord,
} from './datasetExplorerTypes';
import { DatasetId } from './types';

export const RAW_COLUMN_NAMES = [
  'unit_id',
  'cycle',
  'op_setting_1',
  'op_setting_2',
  'op_setting_3',
  ...Array.from({ length: 21 }, (_, i) => `sensor_${i + 1}`),
];

/**
 * Parse an engine index CSV file into structured EngineIndexRecord array.
 */
export function parseEngineIndex(csvText: string): EngineIndexRecord[] {
  if (!csvText || !csvText.trim()) return [];

  const parsed = Papa.parse<Record<string, string>>(csvText.trim(), {
    header: true,
    skipEmptyLines: true,
  });

  if (parsed.errors.length > 0) {
    console.warn('[datasetExplorerParser] CSV parse warnings:', parsed.errors);
  }

  return parsed.data.map((row) => {
    const unitId = Number(row.unit_id);
    const obsCount = Number(row.observation_count);
    const firstCycle = Number(row.first_cycle);
    const lastCycle = Number(row.last_observed_cycle);
    const observedCycleCount = Number(row.observed_cycle_count || row.observation_count);
    const physFail = row.physical_failure_cycle !== '' && row.physical_failure_cycle != null
      ? Number(row.physical_failure_cycle)
      : null;
    const endpointRul = row.official_endpoint_rul !== '' && row.official_endpoint_rul != null
      ? Number(row.official_endpoint_rul)
      : null;

    return {
      dataset: row.dataset as DatasetId,
      split: row.split as 'train' | 'test',
      unit_id: unitId,
      observation_count: obsCount,
      first_cycle: firstCycle,
      last_observed_cycle: lastCycle,
      observed_cycle_count: observedCycleCount,
      endpoint_status: row.endpoint_status as 'run-to-failure' | 'censored',
      physical_failure_cycle: physFail != null && !isNaN(physFail) ? physFail : null,
      official_endpoint_rul: endpointRul != null && !isNaN(endpointRul) ? endpointRul : null,
    };
  });
}

/**
 * Parse raw whitespace-separated C-MAPSS text file (26 columns without header).
 */
export function parseRawObservationText(rawText: string): RawObservationRecord[] {
  if (!rawText || !rawText.trim()) return [];

  const lines = rawText.trim().split(/\r?\n/);
  const records: RawObservationRecord[] = new Array(lines.length);
  let validCount = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const parts = line.split(/\s+/);
    if (parts.length < 26) {
      console.warn(`[datasetExplorerParser] Skipping line ${i + 1}: expected 26 columns, got ${parts.length}`);
      continue;
    }

    const rec: RawObservationRecord = {
      unit_id: parseInt(parts[0], 10),
      cycle: parseInt(parts[1], 10),
      op_setting_1: parseFloat(parts[2]),
      op_setting_2: parseFloat(parts[3]),
      op_setting_3: parseFloat(parts[4]),
      sensor_1: parseFloat(parts[5]),
      sensor_2: parseFloat(parts[6]),
      sensor_3: parseFloat(parts[7]),
      sensor_4: parseFloat(parts[8]),
      sensor_5: parseFloat(parts[9]),
      sensor_6: parseFloat(parts[10]),
      sensor_7: parseFloat(parts[11]),
      sensor_8: parseFloat(parts[12]),
      sensor_9: parseFloat(parts[13]),
      sensor_10: parseFloat(parts[14]),
      sensor_11: parseFloat(parts[15]),
      sensor_12: parseFloat(parts[16]),
      sensor_13: parseFloat(parts[17]),
      sensor_14: parseFloat(parts[18]),
      sensor_15: parseFloat(parts[19]),
      sensor_16: parseFloat(parts[20]),
      sensor_17: parseFloat(parts[21]),
      sensor_18: parseFloat(parts[22]),
      sensor_19: parseFloat(parts[23]),
      sensor_20: parseFloat(parts[24]),
      sensor_21: parseFloat(parts[25]),
    };

    records[validCount++] = rec;
  }

  return records.slice(0, validCount);
}

/**
 * Parse an official RUL text file (one ground-truth integer per line).
 */
export function parseRulVectorText(text: string): number[] {
  if (!text || !text.trim()) return [];
  return text
    .trim()
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => parseInt(l, 10));
}

/**
 * Perform client-side verification on a set of loaded raw observations.
 */
export function validateObservationRecords(records: RawObservationRecord[]): QualityStats {
  if (records.length === 0) {
    return {
      column_count_ok: false,
      column_names_ok: false,
      total_missing_values: 0,
      total_non_finite_values: 0,
      duplicate_rows: 0,
      duplicate_unit_cycle: 0,
      cycles_ordered: false,
      cycles_start_at_1: false,
      valid_unit_ids: false,
      valid_cycle_values: false,
      structural_status: 'FAIL',
      integrity_status: 'FAIL',
    };
  }

  let totalMissing = 0;
  let totalNonFinite = 0;
  let cyclesOrdered = true;
  let cyclesStartAt1 = true;
  let validUnitIds = true;
  let validCycleValues = true;

  const seenUnitCycles = new Set<string>();
  let duplicateUnitCycles = 0;

  // Group by unit_id to check start cycle and ordering
  const unitLastCycle = new Map<number, number>();
  const unitFirstCycle = new Map<number, number>();

  for (let i = 0; i < records.length; i++) {
    const r = records[i];

    if (isNaN(r.unit_id) || r.unit_id <= 0) validUnitIds = false;
    if (isNaN(r.cycle) || r.cycle <= 0) validCycleValues = false;

    // Check numbers in all 26 fields
    for (const key of RAW_COLUMN_NAMES) {
      const val = r[key];
      if (val === undefined || val === null || isNaN(val)) {
        totalMissing++;
      } else if (!isFinite(val)) {
        totalNonFinite++;
      }
    }

    // Check duplicate (unit_id, cycle)
    const ucKey = `${r.unit_id}_${r.cycle}`;
    if (seenUnitCycles.has(ucKey)) {
      duplicateUnitCycles++;
    } else {
      seenUnitCycles.add(ucKey);
    }

    // Check ordering
    const last = unitLastCycle.get(r.unit_id);
    if (last !== undefined) {
      if (r.cycle <= last) {
        cyclesOrdered = false;
      }
    } else {
      unitFirstCycle.set(r.unit_id, r.cycle);
    }
    unitLastCycle.set(r.unit_id, r.cycle);
  }

  for (const first of unitFirstCycle.values()) {
    if (first !== 1) {
      cyclesStartAt1 = false;
      break;
    }
  }

  const structuralStatus = duplicateUnitCycles === 0 ? 'PASS' : 'FAIL';
  const integrityStatus =
    totalMissing === 0 && totalNonFinite === 0 && cyclesOrdered && cyclesStartAt1 && validUnitIds && validCycleValues
      ? 'PASS'
      : 'FAIL';

  return {
    column_count_ok: true,
    column_names_ok: true,
    total_missing_values: totalMissing,
    total_non_finite_values: totalNonFinite,
    duplicate_rows: 0,
    duplicate_unit_cycle: duplicateUnitCycles,
    cycles_ordered: cyclesOrdered,
    cycles_start_at_1: cyclesStartAt1,
    valid_unit_ids: validUnitIds,
    valid_cycle_values: validCycleValues,
    structural_status: structuralStatus,
    integrity_status: integrityStatus,
  };
}
