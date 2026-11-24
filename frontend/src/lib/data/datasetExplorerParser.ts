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

const STRICT_FLOAT_REGEX = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const STRICT_POSITIVE_INT_REGEX = /^[1-9]\d*$/;

/**
 * Strict parser for positive integers (e.g. unit_id, cycle).
 * Rejects negative numbers, zero, floats, and non-numeric suffixes (e.g. "1invalid").
 */
export function parseStrictPositiveInt(token: string, fieldName: string, lineNum: number): number {
  if (!STRICT_POSITIVE_INT_REGEX.test(token)) {
    throw new Error(
      `Parse error on line ${lineNum}: ${fieldName} must be a positive integer, got "${token}"`
    );
  }
  const val = Number(token);
  if (!Number.isSafeInteger(val) || val <= 0) {
    throw new Error(
      `Parse error on line ${lineNum}: ${fieldName} must be a positive integer, got "${token}"`
    );
  }
  return val;
}

/**
 * Strict parser for finite floating point numbers.
 * Rejects NaN, Infinity, -Infinity, empty strings, and non-numeric suffixes (e.g. "12.5invalid").
 */
export function parseStrictFiniteFloat(token: string, fieldName: string, lineNum: number): number {
  if (token === '' || !STRICT_FLOAT_REGEX.test(token)) {
    throw new Error(
      `Parse error on line ${lineNum}: ${fieldName} must be a valid finite number, got "${token}"`
    );
  }
  const val = Number(token);
  if (!Number.isFinite(val)) {
    throw new Error(
      `Parse error on line ${lineNum}: ${fieldName} must be a finite number, got "${token}"`
    );
  }
  return val;
}

/**
 * Parse raw whitespace-separated C-MAPSS text file (strictly 26 columns without header).
 * Fails with descriptive line number and reason on any malformed or corrupted observation.
 */
export function parseRawObservationText(rawText: string): RawObservationRecord[] {
  if (!rawText || !rawText.trim()) return [];

  const lines = rawText.split(/\r?\n/);
  const records: RawObservationRecord[] = [];

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const lineNum = i + 1; // 1-indexed line number in source text
    const line = rawLine.trim();
    if (!line) continue; // Skip empty lines, preserving line count accuracy

    const parts = line.split(/\s+/);
    if (parts.length !== 26) {
      throw new Error(
        `Parse error on line ${lineNum}: expected exactly 26 columns, got ${parts.length}`
      );
    }

    const unit_id = parseStrictPositiveInt(parts[0], 'unit_id', lineNum);
    const cycle = parseStrictPositiveInt(parts[1], 'cycle', lineNum);
    const op_setting_1 = parseStrictFiniteFloat(parts[2], 'op_setting_1', lineNum);
    const op_setting_2 = parseStrictFiniteFloat(parts[3], 'op_setting_2', lineNum);
    const op_setting_3 = parseStrictFiniteFloat(parts[4], 'op_setting_3', lineNum);

    const rec: RawObservationRecord = {
      unit_id,
      cycle,
      op_setting_1,
      op_setting_2,
      op_setting_3,
      sensor_1: parseStrictFiniteFloat(parts[5], 'sensor_1', lineNum),
      sensor_2: parseStrictFiniteFloat(parts[6], 'sensor_2', lineNum),
      sensor_3: parseStrictFiniteFloat(parts[7], 'sensor_3', lineNum),
      sensor_4: parseStrictFiniteFloat(parts[8], 'sensor_4', lineNum),
      sensor_5: parseStrictFiniteFloat(parts[9], 'sensor_5', lineNum),
      sensor_6: parseStrictFiniteFloat(parts[10], 'sensor_6', lineNum),
      sensor_7: parseStrictFiniteFloat(parts[11], 'sensor_7', lineNum),
      sensor_8: parseStrictFiniteFloat(parts[12], 'sensor_8', lineNum),
      sensor_9: parseStrictFiniteFloat(parts[13], 'sensor_9', lineNum),
      sensor_10: parseStrictFiniteFloat(parts[14], 'sensor_10', lineNum),
      sensor_11: parseStrictFiniteFloat(parts[15], 'sensor_11', lineNum),
      sensor_12: parseStrictFiniteFloat(parts[16], 'sensor_12', lineNum),
      sensor_13: parseStrictFiniteFloat(parts[17], 'sensor_13', lineNum),
      sensor_14: parseStrictFiniteFloat(parts[18], 'sensor_14', lineNum),
      sensor_15: parseStrictFiniteFloat(parts[19], 'sensor_15', lineNum),
      sensor_16: parseStrictFiniteFloat(parts[20], 'sensor_16', lineNum),
      sensor_17: parseStrictFiniteFloat(parts[21], 'sensor_17', lineNum),
      sensor_18: parseStrictFiniteFloat(parts[22], 'sensor_18', lineNum),
      sensor_19: parseStrictFiniteFloat(parts[23], 'sensor_19', lineNum),
      sensor_20: parseStrictFiniteFloat(parts[24], 'sensor_20', lineNum),
      sensor_21: parseStrictFiniteFloat(parts[25], 'sensor_21', lineNum),
    };

    records.push(rec);
  }

  return records;
}

/**
 * Parse an official RUL text file (strictly one non-negative ground-truth integer per line).
 */
export function parseRulVectorText(text: string): number[] {
  if (!text || !text.trim()) return [];
  const lines = text.split(/\r?\n/);
  const vector: number[] = [];

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const lineNum = i + 1;
    const line = rawLine.trim();
    if (!line) continue;

    if (!/^(?:0|[1-9]\d*)$/.test(line)) {
      throw new Error(
        `Parse error on line ${lineNum}: RUL value must be a non-negative integer, got "${line}"`
      );
    }
    const val = Number(line);
    if (!Number.isSafeInteger(val) || val < 0) {
      throw new Error(
        `Parse error on line ${lineNum}: RUL value must be a non-negative integer, got "${line}"`
      );
    }
    vector.push(val);
  }

  return vector;
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
