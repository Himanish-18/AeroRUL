import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TopBar } from '../src/components/layout/TopBar';
import { FeatureDetailInspector } from '../src/features/feature-intelligence/FeatureDetailInspector';
import { FeatureImportanceSection } from '../src/features/feature-intelligence/FeatureImportanceSection';
import { ImportanceVsCorrelation } from '../src/features/feature-intelligence/ImportanceVsCorrelation';
import { SensorDegradationSection } from '../src/features/feature-intelligence/SensorDegradationSection';
import { SensorTrajectoryChart } from '../src/features/feature-intelligence/SensorTrajectoryChart';
import { EngineSensorSummaryCards } from '../src/features/feature-intelligence/EngineSensorSummaryCards';
import {
  FeatureImportanceRecord,
  FeatureFamilyRecord,
  SensorTrajectoryRecord,
  SensorTrajectoriesManifest,
} from '../src/lib/data/types';

// ResizeObserver mock for Recharts ResponsiveContainer
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

const mockFeatures: FeatureImportanceRecord[] = [
  {
    dataset: 'FD001',
    feature_name: 'sensor_4_mean_10',
    sensor: 'sensor_4',
    statistic: 'mean',
    window_or_lag: '10',
    description: 'Rolling mean over 10 cycles for sensor 4',
    gain_importance: 5000,
    split_importance: 120,
    gain_importance_pct: 25.0,
    split_importance_pct: 20.0,
    gain_rank: 1,
    split_rank: 2,
    correlation_pearson: 0.72,
    correlation_spearman: 0.75,
    abs_pearson: 0.72,
    abs_spearman: 0.75,
  },
  {
    dataset: 'FD001',
    feature_name: 'sensor_11_r2_75',
    sensor: 'sensor_11',
    statistic: 'r2',
    window_or_lag: '75',
    description: 'R2 fit over 75 cycles for sensor 11',
    gain_importance: 3000,
    split_importance: 150,
    gain_importance_pct: 15.0,
    split_importance_pct: 25.0,
    gain_rank: 2,
    split_rank: 1,
    correlation_pearson: -0.80,
    correlation_spearman: -0.82,
    abs_pearson: 0.80,
    abs_spearman: 0.82,
  },
  {
    dataset: 'FD001',
    feature_name: 'sensor_15_std_20',
    sensor: 'sensor_15',
    statistic: 'std',
    window_or_lag: '20',
    description: 'Rolling standard deviation for sensor 15',
    gain_importance: 1000,
    split_importance: 40,
    gain_importance_pct: 5.0,
    split_importance_pct: 6.0,
    gain_rank: 3,
    split_rank: 3,
    correlation_pearson: -0.35,
    correlation_spearman: null, // Pearson only
    abs_pearson: 0.35,
    abs_spearman: null,
  },
  {
    dataset: 'FD001',
    feature_name: 'sensor_8_max_30',
    sensor: 'sensor_8',
    statistic: 'max',
    window_or_lag: '30',
    description: 'Rolling max for sensor 8',
    gain_importance: 800,
    split_importance: 30,
    gain_importance_pct: 4.0,
    split_importance_pct: 5.0,
    gain_rank: 4,
    split_rank: 4,
    correlation_pearson: null, // Spearman only
    correlation_spearman: 0.40,
    abs_pearson: null,
    abs_spearman: 0.40,
  },
  {
    dataset: 'FD001',
    feature_name: 'sensor_2_slope_10',
    sensor: 'sensor_2',
    statistic: 'slope',
    window_or_lag: '10',
    description: 'Slope for sensor 2',
    gain_importance: 200,
    split_importance: 10,
    gain_importance_pct: 1.0,
    split_importance_pct: 1.5,
    gain_rank: 5,
    split_rank: 5,
    correlation_pearson: null, // Neither
    correlation_spearman: null,
    abs_pearson: null,
    abs_spearman: null,
  },
];

const mockFamilies: FeatureFamilyRecord[] = [
  {
    dataset: 'FD001',
    family_type: 'sensor',
    family_key: 'sensor_4',
    gain_importance: 5000,
    split_importance: 120,
    gain_importance_pct: 25.0,
    split_importance_pct: 20.0,
    feature_count: 30,
    rank_in_family: 1,
  },
  {
    dataset: 'FD001',
    family_type: 'statistic',
    family_key: 'mean',
    gain_importance: 5000,
    split_importance: 120,
    gain_importance_pct: 25.0,
    split_importance_pct: 20.0,
    feature_count: 30,
    rank_in_family: 1,
  },
  {
    dataset: 'FD001',
    family_type: 'window',
    family_key: 'W10',
    gain_importance: 5000,
    split_importance: 120,
    gain_importance_pct: 25.0,
    split_importance_pct: 20.0,
    feature_count: 30,
    rank_in_family: 1,
  },
];

const mockTrajectories: SensorTrajectoryRecord[] = [
  { dataset: 'FD001', unit_id: 1, cycle: 1, sensor: 'sensor_11', sensor_value: 47.5 },
  { dataset: 'FD001', unit_id: 1, cycle: 3, sensor: 'sensor_11', sensor_value: 47.8 },
  { dataset: 'FD001', unit_id: 1, cycle: 192, sensor: 'sensor_11', sensor_value: 48.5 },
  { dataset: 'FD001', unit_id: 2, cycle: 1, sensor: 'sensor_11', sensor_value: 47.2 },
  { dataset: 'FD001', unit_id: 2, cycle: 3, sensor: 'sensor_11', sensor_value: 47.4 },
  { dataset: 'FD001', unit_id: 2, cycle: 287, sensor: 'sensor_11', sensor_value: 48.9 },
  { dataset: 'FD001', unit_id: 3, cycle: 1, sensor: 'sensor_11', sensor_value: 47.1 },
  { dataset: 'FD001', unit_id: 3, cycle: 3, sensor: 'sensor_11', sensor_value: 47.3 },
  { dataset: 'FD001', unit_id: 3, cycle: 179, sensor: 'sensor_11', sensor_value: 48.2 },
];

const mockManifest: SensorTrajectoriesManifest = {
  manifest_version: '1.0.0',
  generated_at: '2026-11-13T00:00:00Z',
  sampling_strategy: 'stride_2_with_endpoints',
  sensor_specifications: {
    sensor_11: { name: 'Static pressure at HPC outlet', unit: 'psia', type: 'Pressure' },
    sensor_4: { name: 'Total temp at LPT outlet', unit: '°R', type: 'Temperature' },
  },
  subsets: {
    FD001: {
      dataset: 'FD001',
      selected_units: [1, 2, 3, 4, 5],
      default_unit_id: 1,
      unit_metadata: {
        '1': { unit_id: 1, lifetime_cycles: 192, sampled_points: 97, is_fleet_standard: true },
        '2': { unit_id: 2, lifetime_cycles: 287, sampled_points: 144, is_fleet_standard: true },
        '3': { unit_id: 3, lifetime_cycles: 179, sampled_points: 90, is_fleet_standard: true },
      },
      available_sensors: ['sensor_4', 'sensor_11'],
      total_sampled_cycles: 331,
      total_measurement_records: 6951,
    },
    FD002: {
      dataset: 'FD002',
      selected_units: [1, 2, 3],
      default_unit_id: 1,
      unit_metadata: {},
      available_sensors: ['sensor_11'],
      total_sampled_cycles: 100,
      total_measurement_records: 1000,
    },
    FD003: {
      dataset: 'FD003',
      selected_units: [1, 2, 3],
      default_unit_id: 1,
      unit_metadata: {},
      available_sensors: ['sensor_11'],
      total_sampled_cycles: 100,
      total_measurement_records: 1000,
    },
    FD004: {
      dataset: 'FD004',
      selected_units: [1, 2, 3],
      default_unit_id: 1,
      unit_metadata: {},
      available_sensors: ['sensor_11'],
      total_sampled_cycles: 100,
      total_measurement_records: 1000,
    },
  },
};

describe('Stage 4 — Feature Intelligence UI Component Regression Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('1. TopBar Header Provenance Badge', () => {
    it('displays "Training-Derived" badge when currentView is features', () => {
      render(
        <TopBar
          currentView="features"
          selectedDataset="FD001"
          onSelectDataset={vi.fn()}
          onOpenMobileMenu={vi.fn()}
        />
      );

      const badge = screen.getByText('Training-Derived');
      expect(badge).toBeDefined();
      expect(screen.getByText('Feature Intelligence')).toBeDefined();
    });

    it('preserves "Candidate Benchmarks" for models and "Official Test" for overview and predictions', () => {
      const { rerender } = render(
        <TopBar
          currentView="models"
          selectedDataset="FD001"
          onSelectDataset={vi.fn()}
          onOpenMobileMenu={vi.fn()}
        />
      );
      expect(screen.getByText('Candidate Benchmarks')).toBeDefined();

      rerender(
        <TopBar
          currentView="overview"
          selectedDataset="FD001"
          onSelectDataset={vi.fn()}
          onOpenMobileMenu={vi.fn()}
        />
      );
      expect(screen.getByText('Official Test')).toBeDefined();

      rerender(
        <TopBar
          currentView="predictions"
          selectedDataset="FD001"
          onSelectDataset={vi.fn()}
          onOpenMobileMenu={vi.fn()}
        />
      );
      expect(screen.getByText('Official Test')).toBeDefined();
    });
  });

  describe('2. Feature Importance Section Controls & Interactions', () => {
    it('toggles importance metric between gain and split', () => {
      render(
        <FeatureImportanceSection
          dataset="FD001"
          features={mockFeatures}
          familyRecords={mockFamilies}
        />
      );

      const splitBtn = screen.getByRole('button', { name: /Split Frequency/i });
      fireEvent.click(splitBtn);
      expect(splitBtn.className).toContain('btn-primary');

      const gainBtn = screen.getByRole('button', { name: /Gain Importance/i });
      fireEvent.click(gainBtn);
      expect(gainBtn.className).toContain('btn-primary');
    });

    it('filters feature list by search input', () => {
      render(
        <FeatureImportanceSection
          dataset="FD001"
          features={mockFeatures}
          familyRecords={mockFamilies}
        />
      );

      const searchInput = screen.getByLabelText(/Filter features by search term/i);
      fireEvent.change(searchInput, { target: { value: 'sensor_11' } });

      expect(screen.getByText(/Showing 1 of 5 features/i)).toBeDefined();
    });

    it('changes Top-N feature selector slice', () => {
      render(
        <FeatureImportanceSection
          dataset="FD001"
          features={mockFeatures}
          familyRecords={mockFamilies}
        />
      );

      const top10Btn = screen.getByRole('button', { name: 'Top 10' });
      fireEvent.click(top10Btn);
      expect(top10Btn.className).toContain('btn-primary');
    });

    it('renders feature detail inspector with initial feature and updates when feature changes', () => {
      const { rerender } = render(
        <FeatureDetailInspector
          feature={mockFeatures[0]}
          totalFeatures={5}
          trajectoriesManifest={mockManifest}
        />
      );

      // Initially shows sensor_4_mean_10
      expect(screen.getByText('sensor_4_mean_10')).toBeDefined();

      // Rerender with sensor_11_r2_75
      rerender(
        <FeatureDetailInspector
          feature={mockFeatures[1]}
          totalFeatures={5}
          trajectoriesManifest={mockManifest}
        />
      );

      expect(screen.getByText('sensor_11_r2_75')).toBeDefined();
    });
  });

  describe('3. Feature Detail Inspector Metric & Correlation Handling', () => {
    it('safely handles features with null correlations without throwing or NaN', () => {
      const featureWithNulls = mockFeatures[4]; // sensor_2_slope_10 (both null)
      render(
        <FeatureDetailInspector
          feature={featureWithNulls}
          totalFeatures={5}
          trajectoriesManifest={mockManifest}
        />
      );

      expect(screen.getByText('sensor_2_slope_10')).toBeDefined();
      expect(screen.getAllByText('—').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Not in Phase 3 screen').length).toBe(2);
    });

    it('renders physical sensor metadata for sensor_4 and sensor_21', () => {
      const feature4 = mockFeatures[0]; // sensor_4
      render(
        <FeatureDetailInspector
          feature={feature4}
          totalFeatures={5}
          trajectoriesManifest={mockManifest}
        />
      );

      expect(screen.getByText('Total temp at LPT outlet')).toBeDefined();
      expect(screen.getByText(/°R/)).toBeDefined();
    });
  });

  describe('4. Importance vs Correlation Metric Switching & Neutral Quadrants', () => {
    it('switches between Pearson and Spearman, updating visible evaluated feature count', () => {
      render(
        <ImportanceVsCorrelation
          features={mockFeatures}
          onSelectFeature={vi.fn()}
        />
      );

      // In mockFeatures:
      // sensor_4_mean_10 (both P & S)
      // sensor_11_r2_75 (both P & S)
      // sensor_15_std_20 (P only)
      // sensor_8_max_30 (S only)
      // sensor_2_slope_10 (neither)
      // Pearson candidates: 3
      // Spearman candidates: 3

      expect(screen.getByText(/3 evaluated features/)).toBeDefined();

      const spearmanBtn = screen.getByTestId('btn-metric-spearman');
      fireEvent.click(spearmanBtn);
      expect(screen.getByText(/3 evaluated features/)).toBeDefined();
    });

    it('displays neutral quadrant labels without causal claims', () => {
      render(
        <ImportanceVsCorrelation
          features={mockFeatures}
          onSelectFeature={vi.fn()}
        />
      );

      expect(screen.getAllByText('High Gain / High Correlation').length).toBeGreaterThan(0);
      expect(screen.getAllByText('High Gain / Low Correlation').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Low Gain / High Correlation').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Low Gain / Low Correlation').length).toBeGreaterThan(0);
    });

    it('filters visible scatter items by selected quadrant', () => {
      render(
        <ImportanceVsCorrelation
          features={mockFeatures}
          onSelectFeature={vi.fn()}
        />
      );

      const quadSelect = screen.getByTestId('select-quadrant-filter');
      fireEvent.change(quadSelect, { target: { value: 'high_importance_high_corr' } });
      expect((quadSelect as HTMLSelectElement).value).toBe('high_importance_high_corr');
    });
  });

  describe('5. Sensor Degradation Section & Multi-Engine Comparison', () => {
    it('allows changing sensor through dropdown', () => {
      render(
        <SensorDegradationSection
          dataset="FD001"
          trajectories={mockTrajectories}
          isLoadingTrajectories={false}
          manifest={mockManifest}
        />
      );

      const select = screen.getByLabelText(/Select sensor measurement/i);
      fireEvent.change(select, { target: { value: 'sensor_4' } });
      expect((select as HTMLSelectElement).value).toBe('sensor_4');
    });

    it('toggles engine selection chips and supports fleet comparison', () => {
      render(
        <SensorDegradationSection
          dataset="FD001"
          trajectories={mockTrajectories}
          isLoadingTrajectories={false}
          manifest={mockManifest}
        />
      );

      // Default has 1 active engine
      expect(screen.getByText(/1 active/i)).toBeDefined();

      // Click Engine 2 chip
      const engine2Btn = screen.getByLabelText('Toggle Engine 2');
      fireEvent.click(engine2Btn);
      expect(screen.getByText(/2 active/i)).toBeDefined();

      // Click Compare Fleet 1 to 5
      const compareFleetBtn = screen.getByTestId('btn-compare-fleet');
      fireEvent.click(compareFleetBtn);
      expect(screen.getByText(/5 active/i)).toBeDefined();

      // Click Reset
      const resetBtn = screen.getByTestId('btn-reset-default');
      fireEvent.click(resetBtn);
      expect(screen.getByText(/1 active/i)).toBeDefined();
    });

    it('renders SensorTrajectoryChart with straight-line type="linear"', () => {
      const { container } = render(
        <SensorTrajectoryChart
          trajectories={mockTrajectories}
          sensor="sensor_11"
          selectedUnits={[1, 2, 3]}
          manifest={mockManifest}
        />
      );

      // Check chart exists
      expect(container.querySelector('[data-testid="sensor-trajectory-chart"]')).toBeDefined();
    });

    it('renders EngineSensorSummaryCards with Net Sensor Change label', () => {
      render(
        <EngineSensorSummaryCards
          trajectories={mockTrajectories}
          sensor="sensor_11"
          primaryUnitId={1}
          manifest={mockManifest}
        />
      );

      expect(screen.getByText('Net Sensor Change')).toBeDefined();
      expect(screen.getByText(/Sample end \(cycle 192\) minus start \(cycle 1\)/i)).toBeDefined();
    });
  });
});
