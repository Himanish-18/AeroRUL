/**
 * AeroRUL — Dataset Explorer Type Definitions & Contracts
 *
 * Strict interfaces for dataset manifests, engine index records,
 * and 26-column raw C-MAPSS observations.
 */

import { DatasetId } from './types';

export type DatasetSplit = 'train' | 'test';

export interface SensorSpecification {
  name: string;
  unit: string;
  type: string;
}

export interface OpSettingSpecification {
  name: string;
  unit: string;
  description: string;
}

export interface TrajectoryStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  p25: number;
  p75: number;
  std: number;
}

export interface OpSettingStats {
  name: string;
  unit: string;
  description: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  n_unique: number;
}

export interface SensorStats {
  name: string;
  unit: string;
  type: string;
  min: number;
  max: number;
  mean: number;
  std: number;
  n_unique: number;
  is_constant: boolean;
  is_near_constant: boolean;
  missing_count: number;
}

export interface QualityStats {
  column_count_ok: boolean;
  column_names_ok: boolean;
  total_missing_values: number;
  total_non_finite_values: number;
  duplicate_rows: number;
  duplicate_unit_cycle: number;
  cycles_ordered: boolean;
  cycles_start_at_1: boolean;
  valid_unit_ids: boolean;
  valid_cycle_values: boolean;
  structural_status: 'PASS' | 'WARNING' | 'FAIL';
  integrity_status: 'PASS' | 'WARNING' | 'FAIL';
}

export interface RulStats {
  type: string;
  description: string;
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
  p25: number;
  p75: number;
  observation_count?: number;
  endpoint_count?: number;
}

export interface SplitSummary {
  row_count: number;
  engine_count: number;
  column_count: number;
  trajectory_statistics: TrajectoryStats;
  operating_settings: Record<string, OpSettingStats>;
  sensors: Record<string, SensorStats>;
  quality: QualityStats;
  rul_statistics: RulStats;
}

export interface DatasetFilesMeta {
  train_filename: string;
  train_sha256: string;
  train_size_bytes: number;
  test_filename: string;
  test_sha256: string;
  test_size_bytes: number;
  rul_filename: string;
  rul_sha256: string;
  rul_size_bytes: number;
}

export interface DatasetProfile {
  dataset: DatasetId;
  conditions: number;
  fault_modes: number;
  description: string;
  files: DatasetFilesMeta;
  train: SplitSummary;
  test: SplitSummary;
}

export interface DatasetManifest {
  manifest_version: string;
  generated_at: string;
  canonical_schema: {
    total_columns: number;
    columns: string[];
    id_columns: string[];
    op_setting_columns: string[];
    sensor_columns: string[];
  };
  sensor_specifications: Record<string, SensorSpecification>;
  operating_setting_specifications: Record<string, OpSettingSpecification>;
  datasets: Record<DatasetId, DatasetProfile>;
  fd004_count_discrepancy_resolution: {
    readme_claimed_train_engines: number;
    readme_claimed_test_engines: number;
    actual_raw_train_engines: number;
    actual_raw_test_engines: number;
    actual_rul_rows: number;
    resolution: string;
  };
}

export interface EngineIndexRecord {
  dataset: DatasetId;
  split: DatasetSplit;
  unit_id: number;
  observation_count: number;
  first_cycle: number;
  last_observed_cycle: number;
  observed_cycle_count: number;
  endpoint_status: 'run-to-failure' | 'censored';
  physical_failure_cycle: number | null;
  official_endpoint_rul: number | null;
}

export interface RawObservationRecord {
  unit_id: number;
  cycle: number;
  op_setting_1: number;
  op_setting_2: number;
  op_setting_3: number;
  sensor_1: number;
  sensor_2: number;
  sensor_3: number;
  sensor_4: number;
  sensor_5: number;
  sensor_6: number;
  sensor_7: number;
  sensor_8: number;
  sensor_9: number;
  sensor_10: number;
  sensor_11: number;
  sensor_12: number;
  sensor_13: number;
  sensor_14: number;
  sensor_15: number;
  sensor_16: number;
  sensor_17: number;
  sensor_18: number;
  sensor_19: number;
  sensor_20: number;
  sensor_21: number;
  [key: string]: number;
}

export interface EngineFilterOptions {
  searchQuery: string;
  sortBy: keyof EngineIndexRecord;
  sortAsc: boolean;
  page: number;
  pageSize: number;
}

export interface ObservationFilterOptions {
  unitId: number;
  minCycle?: number;
  maxCycle?: number;
  selectedSensors: string[];
  sortBy: string;
  sortAsc: boolean;
  page: number;
  pageSize: number;
}
