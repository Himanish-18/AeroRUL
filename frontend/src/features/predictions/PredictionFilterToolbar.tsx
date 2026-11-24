import React from 'react';
import { Search, RotateCcw, Filter } from 'lucide-react';
import { ErrorClassificationFilter, PredictionFilterState } from '../../lib/data/types';

interface PredictionFilterToolbarProps {
  filters: PredictionFilterState;
  onFilterChange: (newFilters: PredictionFilterState) => void;
  totalEngines: number;
  filteredCount: number;
  overCount: number;
  underCount: number;
}

const MIN_ERROR_OPTIONS = [
  { label: 'All error magnitudes', value: 0 },
  { label: '|err| ≥ 10 cycles', value: 10 },
  { label: '|err| ≥ 25 cycles', value: 25 },
  { label: '|err| ≥ 50 cycles', value: 50 },
];

export const PredictionFilterToolbar: React.FC<PredictionFilterToolbarProps> = ({
  filters,
  onFilterChange,
  totalEngines,
  filteredCount,
  overCount,
  underCount,
}) => {
  const isFiltered =
    filters.searchQuery.trim().length > 0 ||
    filters.errorClassification !== 'all' ||
    filters.minAbsoluteError > 0;

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFilterChange({
      ...filters,
      searchQuery: e.target.value,
    });
  };

  const handleClassification = (cls: ErrorClassificationFilter) => {
    onFilterChange({
      ...filters,
      errorClassification: cls,
    });
  };

  const handleMinError = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange({
      ...filters,
      minAbsoluteError: Number(e.target.value),
    });
  };

  const handleReset = () => {
    onFilterChange({
      searchQuery: '',
      errorClassification: 'all',
      minAbsoluteError: 0,
    });
  };

  return (
    <div
      className="panel"
      style={{
        padding: 'var(--space-md) var(--space-lg)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-md)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 'var(--space-md)',
        }}
      >
        {/* Left: Search & Classification */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
          {/* Unit ID Search */}
          <div style={{ position: 'relative', width: '220px' }}>
            <Search
              size={14}
              color="var(--color-muted)"
              style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)' }}
            />
            <input
              type="text"
              className="input-text"
              style={{ width: '100%', paddingLeft: '32px' }}
              placeholder="Filter by Engine ID..."
              value={filters.searchQuery}
              onChange={handleSearch}
              aria-label="Filter by Engine ID"
            />
          </div>

          {/* Classification segmented buttons */}
          <div
            role="group"
            aria-label="Filter by prediction error direction"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '2px',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-pill)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              gap: '2px',
            }}
          >
            <button
              type="button"
              onClick={() => handleClassification('all')}
              aria-pressed={filters.errorClassification === 'all'}
              style={{
                height: '28px',
                padding: '0 var(--space-sm)',
                borderRadius: 'calc(var(--radius-pill) - 2px)',
                fontSize: 'var(--text-xs)',
                fontWeight: filters.errorClassification === 'all' ? 500 : 400,
                background: filters.errorClassification === 'all' ? 'var(--color-accent)' : 'transparent',
                color: filters.errorClassification === 'all' ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              All ({totalEngines})
            </button>
            <button
              type="button"
              onClick={() => handleClassification('over')}
              aria-pressed={filters.errorClassification === 'over'}
              title="Overpredictions: optimistic RUL estimate (predicted RUL > actual RUL)"
              style={{
                height: '28px',
                padding: '0 var(--space-sm)',
                borderRadius: 'calc(var(--radius-pill) - 2px)',
                fontSize: 'var(--text-xs)',
                fontWeight: filters.errorClassification === 'over' ? 500 : 400,
                background: filters.errorClassification === 'over' ? 'var(--color-accent)' : 'transparent',
                color: filters.errorClassification === 'over' ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              Overpredicted ({overCount})
            </button>
            <button
              type="button"
              onClick={() => handleClassification('under')}
              aria-pressed={filters.errorClassification === 'under'}
              title="Underpredictions: conservative RUL estimate (predicted RUL < actual RUL)"
              style={{
                height: '28px',
                padding: '0 var(--space-sm)',
                borderRadius: 'calc(var(--radius-pill) - 2px)',
                fontSize: 'var(--text-xs)',
                fontWeight: filters.errorClassification === 'under' ? 500 : 400,
                background: filters.errorClassification === 'under' ? 'var(--color-accent)' : 'transparent',
                color: filters.errorClassification === 'under' ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              Underpredicted ({underCount})
            </button>
          </div>

          {/* Min error dropdown */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Threshold:</span>
            <select
              className="input-text"
              style={{ height: '32px', paddingRight: '24px', fontSize: 'var(--text-xs)' }}
              value={filters.minAbsoluteError}
              onChange={handleMinError}
              aria-label="Filter by minimum error threshold"
            >
              {MIN_ERROR_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Right: Reset & Filter status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            <Filter size={12} />
            <span>
              Showing <strong style={{ color: 'var(--color-ink)' }}>{filteredCount}</strong> of {totalEngines} engines
            </span>
          </div>

          {isFiltered && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={handleReset}
              style={{
                fontSize: 'var(--text-xs)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
              }}
              aria-label="Reset all filters"
            >
              <RotateCcw size={12} />
              <span>Reset filters</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

