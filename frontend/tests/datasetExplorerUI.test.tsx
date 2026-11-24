/**
 * AeroRUL — Dataset Explorer UI & Component Regression Tests
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DatasetProfileSection } from '../src/features/datasets/DatasetProfileSection';
import { SplitAndEngineExplorer } from '../src/features/datasets/SplitAndEngineExplorer';
import { RawObservationBrowser } from '../src/features/datasets/RawObservationBrowser';
import { SchemaAndQualitySection } from '../src/features/datasets/SchemaAndQualitySection';
import { TopBar } from '../src/components/layout/TopBar';
import { Sidebar } from '../src/components/layout/Sidebar';
import {
  DatasetManifest,
  EngineIndexRecord,
  QualityStats,
  RawObservationRecord,
} from '../src/lib/data/datasetExplorerTypes';

const mockManifest: DatasetManifest = {
  manifest_version: '1.0.0',
  generated_at: '2026-11-23T12:00:00Z',
  canonical_schema: {
    total_columns: 26,
    columns: ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'],
    id_columns: ['unit_id', 'cycle'],
    op_setting_columns: ['op_setting_1', 'op_setting_2', 'op_setting_3'],
    sensor_columns: ['sensor_1', 'sensor_2'],
  },
  sensor_specifications: {
    sensor_1: { name: 'Total temp at fan inlet', unit: '°R', type: 'Temperature' },
    sensor_2: { name: 'Total temp at LPC outlet', unit: '°R', type: 'Temperature' },
    sensor_4: { name: 'Total temp at LPT outlet', unit: '°R', type: 'Temperature' },
  },
  operating_setting_specifications: {
    op_setting_1: { name: 'Altitude', unit: 'kft', description: 'Flight altitude' },
    op_setting_2: { name: 'Mach', unit: 'Mach', description: 'Flight Mach number' },
    op_setting_3: { name: 'TRA', unit: 'TRA', description: 'Throttle resolver angle' },
  },
  datasets: {
    FD001: {
      dataset: 'FD001',
      conditions: 1,
      fault_modes: 1,
      description: 'ONE condition (Sea Level), ONE fault mode (HPC Degradation)',
      files: {
        train_filename: 'train_FD001.txt',
        train_sha256: 'abc1',
        train_size_bytes: 3515356,
        test_filename: 'test_FD001.txt',
        test_sha256: 'abc2',
        test_size_bytes: 2228855,
        rul_filename: 'RUL_FD001.txt',
        rul_sha256: 'abc3',
        rul_size_bytes: 429,
      },
      train: {
        row_count: 20631,
        engine_count: 100,
        column_count: 26,
        trajectory_statistics: {
          min: 128,
          max: 362,
          mean: 206.31,
          median: 199.0,
          p25: 177.0,
          p75: 229.0,
          std: 46.34,
        },
        operating_settings: {},
        sensors: {},
        quality: {
          column_count_ok: true,
          column_names_ok: true,
          total_missing_values: 0,
          total_non_finite_values: 0,
          duplicate_rows: 0,
          duplicate_unit_cycle: 0,
          cycles_ordered: true,
          cycles_start_at_1: true,
          valid_unit_ids: true,
          valid_cycle_values: true,
          structural_status: 'PASS',
          integrity_status: 'PASS',
        },
        rul_statistics: {
          type: 'derived_physical_training_rul',
          description: 'Derived remaining cycles until terminal failure',
          observation_count: 20631,
          min: 0,
          max: 361,
          mean: 107.5,
          median: 99.0,
          std: 68.2,
          p25: 49.0,
          p75: 156.0,
        },
      },
      test: {
        row_count: 13096,
        engine_count: 100,
        column_count: 26,
        trajectory_statistics: {
          min: 31,
          max: 303,
          mean: 130.96,
          median: 133.5,
          p25: 86.0,
          p75: 168.0,
          std: 54.2,
        },
        operating_settings: {},
        sensors: {},
        quality: {
          column_count_ok: true,
          column_names_ok: true,
          total_missing_values: 0,
          total_non_finite_values: 0,
          duplicate_rows: 0,
          duplicate_unit_cycle: 0,
          cycles_ordered: true,
          cycles_start_at_1: true,
          valid_unit_ids: true,
          valid_cycle_values: true,
          structural_status: 'PASS',
          integrity_status: 'PASS',
        },
        rul_statistics: {
          type: 'official_test_endpoint_rul',
          description: 'Official test ground truth endpoint RUL',
          endpoint_count: 100,
          min: 7,
          max: 145,
          mean: 75.52,
          median: 75.0,
          std: 41.2,
          p25: 39.0,
          p75: 112.0,
        },
      },
    },
    FD002: {
      dataset: 'FD002',
      conditions: 6,
      fault_modes: 1,
      description: 'SIX conditions, ONE fault mode (HPC Degradation)',
      files: {} as any,
      train: {
        row_count: 53759,
        engine_count: 260,
        column_count: 26,
        trajectory_statistics: { min: 128, max: 378, mean: 206.77, median: 199.0, p25: 175, p75: 232, std: 45 },
        operating_settings: {},
        sensors: {},
        quality: {} as any,
        rul_statistics: {} as any,
      },
      test: {
        row_count: 33991,
        engine_count: 259,
        column_count: 26,
        trajectory_statistics: { min: 21, max: 367, mean: 131.24, median: 132.0, p25: 85, p75: 169, std: 53 },
        operating_settings: {},
        sensors: {},
        quality: {} as any,
        rul_statistics: { endpoint_count: 259, min: 6, max: 194, mean: 81.19, median: 80, std: 43, p25: 45, p75: 115, type: '', description: '' },
      },
    },
    FD003: {
      dataset: 'FD003',
      conditions: 1,
      fault_modes: 2,
      description: 'ONE condition, TWO fault modes',
      files: {} as any,
      train: { row_count: 24720, engine_count: 100, column_count: 26, trajectory_statistics: { mean: 247.2 } as any } as any,
      test: { row_count: 16596, engine_count: 100, column_count: 26, trajectory_statistics: { mean: 165.96 } as any, rul_statistics: { min: 6, max: 145, mean: 75.32 } as any } as any,
    },
    FD004: {
      dataset: 'FD004',
      conditions: 6,
      fault_modes: 2,
      description: 'SIX conditions, TWO fault modes',
      files: {} as any,
      train: { row_count: 61249, engine_count: 249, column_count: 26, trajectory_statistics: { mean: 245.98 } as any } as any,
      test: { row_count: 41214, engine_count: 248, column_count: 26, trajectory_statistics: { mean: 166.19 } as any, rul_statistics: { min: 6, max: 195, mean: 86.55 } as any } as any,
    },
  },
  fd004_count_discrepancy_resolution: {
    readme_claimed_train_engines: 248,
    readme_claimed_test_engines: 249,
    actual_raw_train_engines: 249,
    actual_raw_test_engines: 248,
    actual_rul_rows: 248,
    resolution: 'CMAPSSData/readme.txt inverted the train and test counts for FD004.',
  },
};

const mockEngines: EngineIndexRecord[] = [
  {
    dataset: 'FD001',
    split: 'train',
    unit_id: 1,
    observation_count: 192,
    first_cycle: 1,
    last_observed_cycle: 192,
    observed_cycle_count: 192,
    endpoint_status: 'run-to-failure',
    physical_failure_cycle: 192,
    official_endpoint_rul: null,
  },
  {
    dataset: 'FD001',
    split: 'train',
    unit_id: 2,
    observation_count: 287,
    first_cycle: 1,
    last_observed_cycle: 287,
    observed_cycle_count: 287,
    endpoint_status: 'run-to-failure',
    physical_failure_cycle: 287,
    official_endpoint_rul: null,
  },
];

const mockObservations: RawObservationRecord[] = [
  {
    unit_id: 1,
    cycle: 1,
    op_setting_1: -0.0007,
    op_setting_2: -0.0004,
    op_setting_3: 100.0,
    sensor_1: 518.67,
    sensor_2: 641.82,
    sensor_3: 1589.7,
    sensor_4: 1400.6,
    sensor_5: 14.62,
    sensor_6: 21.61,
    sensor_7: 554.36,
    sensor_8: 2388.06,
    sensor_9: 9046.19,
    sensor_10: 1.3,
    sensor_11: 47.47,
    sensor_12: 521.66,
    sensor_13: 2388.02,
    sensor_14: 8138.62,
    sensor_15: 8.4195,
    sensor_16: 0.03,
    sensor_17: 392,
    sensor_18: 2388,
    sensor_19: 100,
    sensor_20: 39.06,
    sensor_21: 23.419,
  },
  {
    unit_id: 1,
    cycle: 2,
    op_setting_1: 0.0019,
    op_setting_2: -0.0003,
    op_setting_3: 100.0,
    sensor_1: 518.67,
    sensor_2: 642.15,
    sensor_3: 1591.82,
    sensor_4: 1403.14,
    sensor_5: 14.62,
    sensor_6: 21.61,
    sensor_7: 553.75,
    sensor_8: 2388.04,
    sensor_9: 9044.07,
    sensor_10: 1.3,
    sensor_11: 47.49,
    sensor_12: 522.28,
    sensor_13: 2388.07,
    sensor_14: 8131.49,
    sensor_15: 8.4318,
    sensor_16: 0.03,
    sensor_17: 392,
    sensor_18: 2388,
    sensor_19: 100,
    sensor_20: 39.0,
    sensor_21: 23.4236,
  },
];

describe('Dataset Explorer — UI Component Tests', () => {
  it('renders TopBar with Raw Dataset badge when currentView is datasets', () => {
    render(
      <TopBar
        currentView="datasets"
        selectedDataset="FD001"
        onSelectDataset={vi.fn()}
        onOpenMobileMenu={vi.fn()}
      />
    );

    expect(screen.getByText('Dataset Explorer')).toBeDefined();
    expect(screen.getByText('Raw Dataset')).toBeDefined();
  });

  it('renders Sidebar with active Dataset Explorer item', () => {
    const handleNavigate = vi.fn();
    render(
      <Sidebar
        currentView="overview"
        onNavigate={handleNavigate}
        collapsed={false}
        onToggleCollapse={vi.fn()}
        mobileOpen={false}
        onCloseMobile={vi.fn()}
      />
    );

    const btn = screen.getByText('Dataset Explorer');
    expect(btn).toBeDefined();
    fireEvent.click(btn);
    expect(handleNavigate).toHaveBeenCalledWith('datasets');
  });

  it('renders DatasetProfileSection with fleet metrics and comparison matrix', () => {
    const handleSelectDataset = vi.fn();
    const handleNavigateSplit = vi.fn();

    render(
      <DatasetProfileSection
        manifest={mockManifest}
        selectedDataset="FD001"
        onSelectDataset={handleSelectDataset}
        onNavigateToSplit={handleNavigateSplit}
      />
    );

    // Fleet numbers
    expect(screen.getAllByText('FD001').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('20,631')).toBeDefined();
    expect(screen.getByText('13,096')).toBeDefined();

    // Click comparison row for FD002
    const fd002Row = screen.getByLabelText('Select dataset FD002');
    fireEvent.click(fd002Row);
    expect(handleSelectDataset).toHaveBeenCalledWith('FD002');

    // Click inspect train
    const inspectTrainBtn = screen.getByText('Inspect training fleet');
    fireEvent.click(inspectTrainBtn);
    expect(handleNavigateSplit).toHaveBeenCalledWith('train');
  });

  it('renders SplitAndEngineExplorer with engine index table and search filter', () => {
    const handleSelectSplit = vi.fn();
    const handleSelectEngine = vi.fn();
    const handleInspectRaw = vi.fn();

    render(
      <SplitAndEngineExplorer
        dataset="FD001"
        split="train"
        onSelectSplit={handleSelectSplit}
        engineIndex={mockEngines}
        selectedUnitId={1}
        onSelectEngine={handleSelectEngine}
        onInspectEngineInRaw={handleInspectRaw}
      />
    );

    expect(screen.getByText('Unit #1')).toBeDefined();
    expect(screen.getByText('Unit #2')).toBeDefined();

    // Filter by search
    const searchInput = screen.getByPlaceholderText('Search engine #…');
    fireEvent.change(searchInput, { target: { value: '2' } });
    expect(screen.queryByText('Unit #1')).toBeNull();
    expect(screen.getByText('Unit #2')).toBeDefined();

    // Click inspect in raw
    const inspectRawBtn = screen.getByText(/Inspect raw cycles/i);
    fireEvent.click(inspectRawBtn);
    expect(handleInspectRaw).toHaveBeenCalledWith(1);
  });

  it('renders RawObservationBrowser with authentic C-MAPSS rows and cycle filters', () => {
    const handleSelectSplit = vi.fn();
    const handleSelectUnit = vi.fn();

    render(
      <RawObservationBrowser
        dataset="FD001"
        split="train"
        onSelectSplit={handleSelectSplit}
        availableUnits={[1, 2]}
        selectedUnitId={1}
        onSelectUnit={handleSelectUnit}
        rawObservations={mockObservations}
        isLoading={false}
        sensorSpecs={mockManifest.sensor_specifications}
      />
    );

    expect(screen.getByText('Raw C-MAPSS observation')).toBeDefined();
    expect(screen.getByText('641.8200')).toBeDefined(); // cycle 1 sensor 2
    expect(screen.getByText('642.1500')).toBeDefined(); // cycle 2 sensor 2

    // Change unit in dropdown
    const unitSelect = screen.getByLabelText('Select Engine Unit');
    fireEvent.change(unitSelect, { target: { value: '2' } });
    expect(handleSelectUnit).toHaveBeenCalledWith(2);

    // Filter cycle range
    const minInput = screen.getByLabelText('Minimum cycle filter');
    fireEvent.change(minInput, { target: { value: '2' } });
    expect(screen.queryByText('641.8200')).toBeNull(); // cycle 1 filtered out
    expect(screen.getByText('642.1500')).toBeDefined(); // cycle 2 remains
  });

  it('renders SchemaAndQualitySection with all validation checks passing', () => {
    const quality: QualityStats = {
      column_count_ok: true,
      column_names_ok: true,
      total_missing_values: 0,
      total_non_finite_values: 0,
      duplicate_rows: 0,
      duplicate_unit_cycle: 0,
      cycles_ordered: true,
      cycles_start_at_1: true,
      valid_unit_ids: true,
      valid_cycle_values: true,
      structural_status: 'PASS',
      integrity_status: 'PASS',
    };

    render(
      <SchemaAndQualitySection
        dataset="FD001"
        split="train"
        qualityStats={quality}
        sensorSpecs={mockManifest.sensor_specifications}
        opSettingSpecs={mockManifest.operating_setting_specifications}
      />
    );

    expect(screen.getByText('Data Integrity & Quality Audit (FD001 train)')).toBeDefined();
    expect(screen.getAllByText('PASS').length).toBeGreaterThan(5);
    expect(screen.getByText('C-MAPSS Canonical 26-Column Variable Dictionary')).toBeDefined();
  });

  it('renders SchemaAndQualitySection with failing checks displaying red FAIL badges', () => {
    const failingQuality: QualityStats = {
      column_count_ok: false,
      column_names_ok: false,
      total_missing_values: 5,
      total_non_finite_values: 2,
      duplicate_rows: 3,
      duplicate_unit_cycle: 1,
      cycles_ordered: false,
      cycles_start_at_1: false,
      valid_unit_ids: false,
      valid_cycle_values: false,
      structural_status: 'FAIL',
      integrity_status: 'FAIL',
    };

    render(
      <SchemaAndQualitySection
        dataset="FD001"
        split="train"
        qualityStats={failingQuality}
        sensorSpecs={mockManifest.sensor_specifications}
        opSettingSpecs={mockManifest.operating_setting_specifications}
      />
    );

    expect(screen.getByTestId('badge-structural-status').textContent).toContain('Structural: FAIL');
    expect(screen.getByTestId('badge-structural-status').className).toContain('chip-bad');
    expect(screen.getByTestId('badge-integrity-status').textContent).toContain('Integrity: FAIL');
    expect(screen.getByTestId('badge-integrity-status').className).toContain('chip-bad');

    expect(screen.getByTestId('check-schema-width-fail')).toBeDefined();
    expect(screen.getByTestId('check-missing-values-fail')).toBeDefined();
    expect(screen.getByTestId('check-non-finite-fail')).toBeDefined();
    expect(screen.getByTestId('check-duplicate-rows-fail')).toBeDefined();
    expect(screen.getByTestId('check-duplicate-unit-cycle-fail')).toBeDefined();
    expect(screen.getByTestId('check-cycles-ordered-fail')).toBeDefined();
    expect(screen.getByTestId('check-cycles-start-at-1-fail')).toBeDefined();
    expect(screen.getByTestId('check-valid-unit-ids-fail')).toBeDefined();
  });

  it('supports keyboard-accessible engine selection and column sorting in SplitAndEngineExplorer', () => {
    const handleSelectEngine = vi.fn();

    render(
      <SplitAndEngineExplorer
        dataset="FD001"
        split="train"
        onSelectSplit={vi.fn()}
        engineIndex={mockEngines}
        selectedUnitId={1}
        onSelectEngine={handleSelectEngine}
        onInspectEngineInRaw={vi.fn()}
      />
    );

    // Verify row does NOT have role="button" (table semantics preserved)
    const tableRows = screen.getAllByRole('row');
    for (const r of tableRows) {
      expect(r.getAttribute('role')).toBeNull();
    }

    // Verify keyboard-accessible engine selection button inside the cell
    const selectUnit2Btn = screen.getByRole('button', { name: 'Select Engine Unit #2' });
    expect(selectUnit2Btn).toBeDefined();
    fireEvent.click(selectUnit2Btn);
    expect(handleSelectEngine).toHaveBeenCalledWith(2);

    // Check table headers have aria-sort and can be sorted by button click
    const unitIdHeaderBtn = screen.getByRole('button', { name: /Sort by Unit ID/i });
    expect(unitIdHeaderBtn).toBeDefined();
    const thUnitId = unitIdHeaderBtn.closest('th');
    expect(thUnitId?.getAttribute('aria-sort')).toBe('ascending');

    // Click sort header to toggle descending
    fireEvent.click(unitIdHeaderBtn);
    expect(thUnitId?.getAttribute('aria-sort')).toBe('descending');
  });

  it('supports keyboard-accessible header sorting and aria-sort in RawObservationBrowser', () => {
    render(
      <RawObservationBrowser
        dataset="FD001"
        split="train"
        onSelectSplit={vi.fn()}
        availableUnits={[1, 2]}
        selectedUnitId={1}
        onSelectUnit={vi.fn()}
        rawObservations={mockObservations}
        isLoading={false}
        sensorSpecs={mockManifest.sensor_specifications}
      />
    );

    const cycleSortBtn = screen.getByRole('button', { name: /Sort by cycle/i });
    const thCycle = cycleSortBtn.closest('th');
    expect(thCycle?.getAttribute('aria-sort')).toBe('ascending');

    fireEvent.click(cycleSortBtn);
    expect(thCycle?.getAttribute('aria-sort')).toBe('descending');
  });

  it('displays visible error alert and retry button when engine index fetch fails in DatasetExplorerPage', async () => {
    const { DatasetExplorerPage } = await import('../src/features/datasets/DatasetExplorerPage');
    const loader = await import('../src/lib/data/datasetExplorerLoader');

    vi.spyOn(loader, 'loadDatasetManifest').mockResolvedValue(mockManifest);
    const indexSpy = vi.spyOn(loader, 'loadEngineIndex').mockRejectedValueOnce(
      new Error('Simulated network timeout on index CSV')
    );

    render(<DatasetExplorerPage selectedDataset="FD001" />);

    // Switch to Engines tab
    const enginesTab = await screen.findByRole('tab', { name: /Engines/i });
    fireEvent.click(enginesTab);

    // Verify error alert is rendered
    const alert = await screen.findByRole('alert');
    expect(alert).toBeDefined();
    expect(screen.getByText('Engine Index Load Error')).toBeDefined();
    expect(screen.getByText(/Simulated network timeout on index CSV/)).toBeDefined();

    // Verify retry button exists
    const retryBtn = screen.getByRole('button', { name: /Retry engine index/i });
    expect(retryBtn).toBeDefined();

    // Now mock resolves successfully for the retry
    indexSpy.mockResolvedValueOnce(mockEngines);
    fireEvent.click(retryBtn);

    // Verify engine table successfully displays
    expect(await screen.findByText('Training Fleet Engines (2 matching)')).toBeDefined();
  }, 15000);
});
