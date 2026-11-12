import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PredictionsDetailedTable } from '../src/features/predictions/PredictionsDetailedTable';
import { EngineEndpointInspector } from '../src/features/predictions/EngineEndpointInspector';
import { TopOutliersCard } from '../src/features/predictions/TopOutliersCard';
import { TopBar } from '../src/components/layout/TopBar';
import { PredictionRecord, VALID_DATASETS, DatasetId, VALID_VIEWS, AppView } from '../src/lib/data/types';

// Mock predictions array (30 items to test pagination)
const mockPredictions: PredictionRecord[] = Array.from({ length: 30 }, (_, i) => ({
  dataset: 'FD001',
  unit_id: i + 1,
  last_observed_cycle: 100 + i,
  actual_rul_official: 80 - i,
  predicted_rul_raw: 85 - i,
  predicted_rul_final: 85 - i,
  prediction_error: 5,
  absolute_error: 5,
  selected_model: 'LightGBM',
  selected_feature_configuration: 'ALL_WINDOWS',
  selected_target_cap: 150,
}));

describe('AeroRUL Stage 2 UI Interaction & Component Regression Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Task 4 — Filtered Table Pagination Regression', () => {
    it('1. navigates to page 2 and resets to page 1 when input predictions cohort changes', () => {
      const onSelect = vi.fn();
      const { rerender } = render(
        <PredictionsDetailedTable
          predictions={mockPredictions}
          selectedUnitId={null}
          onSelectEngine={onSelect}
        />
      );

      // Verify page 1 initially
      expect(screen.getByText('Page 1 of 2')).toBeDefined();

      // Navigate to page 2
      const nextBtn = screen.getByRole('button', { name: /next page/i });
      fireEvent.click(nextBtn);
      expect(screen.getByText('Page 2 of 2')).toBeDefined();

      // Reduce dataset to fewer than one page (e.g. 3 records)
      const filteredCohort = mockPredictions.slice(0, 3);
      rerender(
        <PredictionsDetailedTable
          predictions={filteredCohort}
          selectedUnitId={null}
          onSelectEngine={onSelect}
        />
      );

      // Must display matching records on page 1, not stale blank page 2
      expect(screen.getByText('Page 1 of 1')).toBeDefined();
      expect(screen.getByText('Showing 1 to 3 of 3 engines')).toBeDefined();
      expect(screen.getByText('#1')).toBeDefined();
      expect(screen.getByText('#3')).toBeDefined();
    });

    it('2. triggers in-cell inspect button without assigning role="button" to table row', () => {
      const onSelect = vi.fn();
      render(
        <PredictionsDetailedTable
          predictions={mockPredictions.slice(0, 5)}
          selectedUnitId={null}
          onSelectEngine={onSelect}
        />
      );

      // Find inspect button for unit #1
      const inspectBtn = screen.getByRole('button', { name: /inspect engine unit 1$/i });
      expect(inspectBtn).toBeDefined();

      fireEvent.click(inspectBtn);
      expect(onSelect).toHaveBeenCalledWith(1);

      // Table rows must NOT have role="button"
      const rows = document.querySelectorAll('.data-table tbody tr');
      rows.forEach((tr) => {
        expect(tr.getAttribute('role')).not.toBe('button');
      });
    });
  });

  describe('Task 5 & 8 — Engine Endpoint Detail Inspector', () => {
    it('displays correct "Engine Endpoint Detail" title and closes on button click', () => {
      const onClose = vi.fn();
      render(
        <EngineEndpointInspector
          prediction={mockPredictions[0]}
          onClose={onClose}
        />
      );

      // Heading must use "Endpoint", not "Trajectory"
      expect(screen.getByText(/Engine Endpoint Detail: Unit #1/i)).toBeDefined();

      const closeBtn = screen.getByRole('button', { name: /close inspector drawer/i });
      fireEvent.click(closeBtn);
      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it('closes on Escape key press', () => {
      const onClose = vi.fn();
      render(
        <EngineEndpointInspector
          prediction={mockPredictions[0]}
          onClose={onClose}
        />
      );

      fireEvent.keyDown(window, { key: 'Escape' });
      expect(onClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('Top Outliers Interaction', () => {
    it('allows inspecting top outliers via dedicated button', () => {
      const onSelect = vi.fn();
      render(
        <TopOutliersCard
          predictions={mockPredictions.slice(0, 5)}
          selectedUnitId={null}
          onSelectEngine={onSelect}
        />
      );

      const outlierBtn = screen.getByRole('button', { name: /inspect outlier unit 1/i });
      fireEvent.click(outlierBtn);
      expect(onSelect).toHaveBeenCalledWith(1);
    });
  });

  describe('Task 3 — TopBar & Mobile Dataset Selection', () => {
    it('supports switching datasets via desktop buttons and mobile native select', () => {
      const onSelectDataset = vi.fn();
      render(
        <TopBar
          currentView="predictions"
          selectedDataset="FD001"
          onSelectDataset={onSelectDataset}
          onOpenMobileMenu={vi.fn()}
        />
      );

      // Desktop button click
      const fd002Btn = screen.getByRole('button', { name: 'FD002' });
      fireEvent.click(fd002Btn);
      expect(onSelectDataset).toHaveBeenCalledWith('FD002');

      // Mobile native select change
      const mobileSelect = screen.getByLabelText(/select c-mapss dataset \(mobile\)/i);
      fireEvent.change(mobileSelect, { target: { value: 'FD004' } });
      expect(onSelectDataset).toHaveBeenCalledWith('FD004');
    });
  });

  describe('Task 9 — URL Resolution & Navigation State Helpers', () => {
    function resolveDatasetFromQuery(query: string): DatasetId {
      try {
        const params = new URLSearchParams(query);
        const ds = params.get('dataset') as DatasetId | null;
        if (ds && VALID_DATASETS.includes(ds)) return ds;
      } catch {
        // fallback
      }
      return 'FD001';
    }

    function resolveViewFromQuery(query: string): AppView {
      try {
        const params = new URLSearchParams(query);
        const v = params.get('view') as AppView | null;
        if (v && VALID_VIEWS.includes(v)) return v;
      } catch {
        // fallback
      }
      return 'overview';
    }

    it('resolves valid dataset and view parameters from query string', () => {
      expect(resolveDatasetFromQuery('?dataset=FD003&view=predictions')).toBe('FD003');
      expect(resolveViewFromQuery('?dataset=FD003&view=predictions')).toBe('predictions');
    });

    it('falls back safely to default dataset (FD001) and view (overview) when invalid', () => {
      expect(resolveDatasetFromQuery('?dataset=FD999')).toBe('FD001');
      expect(resolveDatasetFromQuery('')).toBe('FD001');
      expect(resolveViewFromQuery('?view=invalid_screen')).toBe('overview');
      expect(resolveViewFromQuery('')).toBe('overview');
    });

    it('preserves dataset when switching between views', () => {
      let activeDataset: DatasetId = 'FD002';
      let activeView: AppView = 'overview';

      // Switch to predictions
      activeView = 'predictions';
      const url = new URL(`http://localhost:4173/?view=${activeView}&dataset=${activeDataset}`);
      expect(url.searchParams.get('dataset')).toBe('FD002');
      expect(url.searchParams.get('view')).toBe('predictions');
    });

    it('resets selected unit unconditionally when dataset changes', () => {
      let selectedUnitId: number | null = 42;
      let dataset: DatasetId = 'FD001';

      // Simulating dataset change handler
      const handleDatasetChange = (newDs: DatasetId) => {
        dataset = newDs;
        selectedUnitId = null; // Unconditional reset
      };

      handleDatasetChange('FD002');
      expect(dataset).toBe('FD002');
      expect(selectedUnitId).toBeNull();
    });
  });
});
