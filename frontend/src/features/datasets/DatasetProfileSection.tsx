import React from 'react';
import {
  DatasetManifest,
  DatasetSplit,
} from '../../lib/data/datasetExplorerTypes';
import {
  getDatasetComparisonRows,
} from '../../lib/data/datasetExplorerAdapter';
import { DatasetId } from '../../lib/data/types';
import { Info, AlertCircle } from 'lucide-react';

interface DatasetProfileSectionProps {
  manifest: DatasetManifest;
  selectedDataset: DatasetId;
  onSelectDataset: (id: DatasetId) => void;
  onNavigateToSplit: (split: DatasetSplit) => void;
}

export const DatasetProfileSection: React.FC<DatasetProfileSectionProps> = ({
  manifest,
  selectedDataset,
  onSelectDataset,
  onNavigateToSplit,
}) => {
  const profile = manifest.datasets[selectedDataset];
  const train = profile.train;
  const test = profile.test;
  const comparisonRows = getDatasetComparisonRows(manifest);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Top Overview Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: 'var(--space-md)',
        }}
      >
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Dataset Profile</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-sm)' }}>
            <span style={{ fontSize: 'var(--text-2xl)', fontWeight: 600, color: 'var(--color-accent)' }}>
              {selectedDataset}
            </span>
            <span className="chip" style={{ fontSize: 'var(--text-xs)' }}>
              {profile.conditions} {profile.conditions === 1 ? 'Condition' : 'Conditions'} · {profile.fault_modes} {profile.fault_modes === 1 ? 'Fault Mode' : 'Fault Modes'}
            </span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
            {profile.description}
          </span>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Training Fleet (Run-to-failure)</span>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => onNavigateToSplit('train')}
              style={{ fontSize: 'var(--text-xs)', whiteSpace: 'nowrap' }}
            >
              Inspect training fleet
            </button>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-sm)' }}>
            <span className="measured" style={{ fontSize: 'var(--text-2xl)', fontWeight: 600, color: 'var(--color-ink)' }}>
              {train.engine_count}
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>engines</span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            <span className="measured">{train.row_count.toLocaleString()}</span> observations · Avg length <span className="measured">{train.trajectory_statistics.mean}</span> cycles (min {train.trajectory_statistics.min}, max {train.trajectory_statistics.max})
          </span>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Test Fleet (Censored)</span>
            <button
              type="button"
              className="btn btn-ghost btn-xs"
              onClick={() => onNavigateToSplit('test')}
              style={{ fontSize: 'var(--text-xs)', whiteSpace: 'nowrap' }}
            >
              Inspect test fleet
            </button>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-sm)' }}>
            <span className="measured" style={{ fontSize: 'var(--text-2xl)', fontWeight: 600, color: 'var(--color-ink)' }}>
              {test.engine_count}
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>engines</span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            <span className="measured">{test.row_count.toLocaleString()}</span> observations · Avg observed <span className="measured">{test.trajectory_statistics.mean}</span> cycles (min {test.trajectory_statistics.min}, max {test.trajectory_statistics.max})
          </span>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-xs)' }}>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Official Test Endpoint RUL</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-sm)' }}>
            <span className="measured" style={{ fontSize: 'var(--text-2xl)', fontWeight: 600, color: 'var(--color-accent)' }}>
              {test.rul_statistics.mean}
            </span>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>mean cycles</span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            {test.rul_statistics.endpoint_count} ground-truth labels · Range <span className="measured">{test.rul_statistics.min} to {test.rul_statistics.max}</span> cycles (median {test.rul_statistics.median})
          </span>
        </div>
      </div>

      {/* Dataset Comparison Table */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h3 style={{ fontSize: 'var(--text-md)', color: 'var(--color-ink)', margin: 0 }}>
              C-MAPSS Benchmark Subsets Matrix (FD001 to FD004)
            </h3>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Select any row to switch active dataset. Verified against committed raw files.
            </span>
          </div>
        </div>

        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <table className="table" style={{ width: '100%', fontSize: 'var(--text-xs)', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: 'var(--rule-hair) solid var(--color-rule-2)', textAlign: 'left' }}>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)' }}>Dataset</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)' }}>Conditions</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)' }}>Fault Modes</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)', textAlign: 'right' }}>Train Engines / Rows</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)', textAlign: 'right' }}>Test Engines / Rows</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)', textAlign: 'right' }}>Avg Train Lifetime</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)', textAlign: 'right' }}>Avg Test Observed</th>
                <th style={{ padding: '8px var(--space-sm)', color: 'var(--color-muted)', textAlign: 'right' }}>Test RUL Range</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row) => {
                const isSelected = row.dataset === selectedDataset;
                return (
                  <tr
                    key={row.dataset}
                    onClick={() => onSelectDataset(row.dataset)}
                    style={{
                      cursor: 'pointer',
                      background: isSelected ? 'var(--color-paper-3)' : 'transparent',
                      borderBottom: 'var(--rule-hair) solid var(--color-rule)',
                      fontWeight: isSelected ? 600 : 400,
                    }}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        onSelectDataset(row.dataset);
                      }
                    }}
                    aria-label={`Select dataset ${row.dataset}`}
                  >
                    <td style={{ padding: '8px var(--space-sm)' }}>
                      <span
                        style={{
                          color: isSelected ? 'var(--color-accent)' : 'var(--color-ink)',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                        }}
                      >
                        {isSelected && <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--color-accent)' }} />}
                        {row.dataset}
                      </span>
                    </td>
                    <td style={{ padding: '8px var(--space-sm)', color: 'var(--color-ink-2)' }}>
                      {row.conditions} {row.conditions === 1 ? '(Sea Level)' : '(Six Regimes)'}
                    </td>
                    <td style={{ padding: '8px var(--space-sm)', color: 'var(--color-ink-2)' }}>
                      {row.faultModesText}
                    </td>
                    <td className="measured" style={{ padding: '8px var(--space-sm)', textAlign: 'right', color: 'var(--color-ink)' }}>
                      {row.trainEngines} / {row.trainRows.toLocaleString()}
                    </td>
                    <td className="measured" style={{ padding: '8px var(--space-sm)', textAlign: 'right', color: 'var(--color-ink)' }}>
                      {row.testEngines} / {row.testRows.toLocaleString()}
                    </td>
                    <td className="measured" style={{ padding: '8px var(--space-sm)', textAlign: 'right', color: 'var(--color-ink)' }}>
                      {row.avgTrainLength} cycles
                    </td>
                    <td className="measured" style={{ padding: '8px var(--space-sm)', textAlign: 'right', color: 'var(--color-ink-2)' }}>
                      {row.avgTestLength} cycles
                    </td>
                    <td className="measured" style={{ padding: '8px var(--space-sm)', textAlign: 'right', color: 'var(--color-accent)' }}>
                      {row.rulRangeText}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Critical Semantic Distinctions & Discrepancy Note */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: 'var(--space-md)',
        }}
      >
        <div
          className="panel"
          style={{
            background: 'var(--color-paper-2)',
            borderLeft: '3px solid var(--color-cool)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-xs)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', color: 'var(--color-cool)' }}>
            <Info size={16} />
            <span style={{ fontWeight: 600, fontSize: 'var(--text-xs)' }}>Critical Data Semantics</span>
          </div>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', margin: 0, lineHeight: 1.5 }}>
            <strong>Training trajectories</strong> are complete run-to-failure records. The terminal observed cycle is the physical failure point, allowing exact derivation of physical RUL at each past cycle without censoring.
          </p>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', margin: 0, lineHeight: 1.5 }}>
            <strong>Test trajectories</strong> are censored before failure. The official RUL ground-truth vector contains exactly one label per test engine, applicable exclusively to its terminal recorded observation cycle. Test endpoint cycles must never be interpreted as failure points.
          </p>
        </div>

        <div
          className="panel"
          style={{
            background: 'var(--color-paper-2)',
            borderLeft: '3px solid var(--color-accent)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-xs)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', color: 'var(--color-accent)' }}>
            <AlertCircle size={16} />
            <span style={{ fontWeight: 600, fontSize: 'var(--text-xs)' }}>FD004 Fleet Count Reconciliation</span>
          </div>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', margin: 0, lineHeight: 1.5 }}>
            {manifest.fd004_count_discrepancy_resolution.resolution}
          </p>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            Verified directly from raw files: <code>train_FD004.txt</code> (249 units), <code>test_FD004.txt</code> (248 units), <code>RUL_FD004.txt</code> (248 rows).
          </span>
        </div>
      </div>
    </div>
  );
};
