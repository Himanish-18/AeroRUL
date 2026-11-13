import React from 'react';
import { Menu, ShieldCheck, Database } from 'lucide-react';
import { DatasetId, VALID_DATASETS, AppView } from '../../lib/data/types';

interface TopBarProps {
  currentView: AppView;
  selectedDataset: DatasetId;
  onSelectDataset: (dataset: DatasetId) => void;
  frozenTimestamp?: string;
  onOpenMobileMenu: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentView,
  selectedDataset,
  onSelectDataset,
  frozenTimestamp,
  onOpenMobileMenu,
}) => {
  const isPredictions = currentView === 'predictions';

  return (
    <header className="app-header">
      {/* Left: Mobile trigger & context */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
        <button
          type="button"
          className="btn btn-icon btn-sm mobile-menu-btn"
          onClick={onOpenMobileMenu}
          aria-label="Open navigation menu"
          id="mobile-menu-btn"
        >
          <Menu size={18} />
        </button>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <span
              style={{
                fontSize: 'var(--text-sm)',
                fontWeight: 500,
                color: 'var(--color-ink)',
                letterSpacing: '-0.01em',
              }}
            >
              {currentView === 'models'
                ? 'Model Evaluation'
                : currentView === 'features'
                ? 'Feature Intelligence'
                : isPredictions
                ? 'Engine Predictions'
                : 'Evaluation overview'}
            </span>
            <span
              className="header-badge"
              style={{
                fontSize: '0.65rem',
                color: 'var(--color-muted)',
                padding: '1px 6px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--color-paper-3)',
                border: 'var(--rule-hair) solid var(--color-rule)',
              }}
            >
              {currentView === 'models' ? 'Candidate Benchmarks' : 'Official Test'}
            </span>
          </div>
          <span className="header-subtitle" style={{ fontSize: '0.75rem', color: 'var(--color-muted)', lineHeight: 1.2 }}>
            {currentView === 'models'
              ? 'Candidate model benchmarking, cross-validation, and selection evidence'
              : isPredictions
              ? 'Official test engine RUL endpoints and error decomposition'
              : 'NASA C-MAPSS turbofan degradation benchmark'}
          </span>
        </div>
      </div>

      {/* Right: Dataset selector & frozen timestamp */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
        {/* Mobile compact native select (< 40rem) */}
        <div className="dataset-mobile-selector">
          <Database size={14} color="var(--color-muted)" />
          <select
            className="dataset-native-select"
            value={selectedDataset}
            onChange={(e) => onSelectDataset(e.target.value as DatasetId)}
            aria-label="Select C-MAPSS dataset (mobile)"
          >
            {VALID_DATASETS.map((ds) => (
              <option key={ds} value={ds}>
                {ds}
              </option>
            ))}
          </select>
        </div>

        {/* Desktop / Tablet segmented button pills (>= 40rem) */}
        <div
          role="group"
          aria-label="Select C-MAPSS dataset subset"
          className="dataset-desktop-selector"
          style={{
            alignItems: 'center',
            padding: '2px',
            background: 'var(--color-paper-3)',
            borderRadius: 'var(--radius-pill)',
            border: 'var(--rule-hair) solid var(--color-rule)',
            gap: '2px',
          }}
        >
          <span
            style={{
              padding: '0 var(--space-xs)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-muted)',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Database size={12} />
            <span className="dataset-label-text">Dataset</span>
          </span>
          {VALID_DATASETS.map((ds) => {
            const isSelected = ds === selectedDataset;
            return (
              <button
                key={ds}
                type="button"
                onClick={() => onSelectDataset(ds)}
                aria-pressed={isSelected}
                style={{
                  height: '28px',
                  padding: '0 var(--space-sm)',
                  borderRadius: 'calc(var(--radius-pill) - 2px)',
                  fontSize: 'var(--text-xs)',
                  fontFamily: 'var(--font-outlier)',
                  fontWeight: isSelected ? 500 : 400,
                  background: isSelected ? 'var(--color-accent)' : 'transparent',
                  color: isSelected ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                  border: 'none',
                  transition: 'background-color var(--dur-micro) var(--ease-out), color var(--dur-micro) var(--ease-out)',
                  cursor: 'pointer',
                }}
              >
                {ds}
              </button>
            );
          })}
        </div>

        {/* Frozen Manifest Stamp */}
        {frozenTimestamp && (
          <div
            className="header-stamp"
            title={`Frozen model manifest timestamp: ${frozenTimestamp}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-2xs)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-muted)',
              fontFamily: 'var(--font-outlier)',
              padding: '4px var(--space-xs)',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--color-paper-3)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <ShieldCheck size={14} color="var(--color-cool)" />
            <span>{frozenTimestamp.split(' ')[0]}</span>
          </div>
        )}
      </div>
    </header>
  );
};

