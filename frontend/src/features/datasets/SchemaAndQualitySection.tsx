import React, { useState } from 'react';
import {
  DatasetSplit,
  QualityStats,
  SensorSpecification,
  OpSettingSpecification,
} from '../../lib/data/datasetExplorerTypes';
import { RAW_COLUMN_NAMES } from '../../lib/data/datasetExplorerParser';
import { DatasetId } from '../../lib/data/types';
import { CheckCircle2, AlertTriangle, Search, ShieldCheck } from 'lucide-react';

interface SchemaAndQualitySectionProps {
  dataset: DatasetId;
  split: DatasetSplit;
  qualityStats: QualityStats;
  sensorSpecs: Record<string, SensorSpecification>;
  opSettingSpecs: Record<string, OpSettingSpecification>;
}

export const SchemaAndQualitySection: React.FC<SchemaAndQualitySectionProps> = ({
  dataset,
  split,
  qualityStats,
  sensorSpecs,
  opSettingSpecs,
}) => {
  const [searchFilter, setSearchFilter] = useState('');

  // Helper to render dynamic status chips (green PASS vs red FAIL)
  const renderQualityBadge = (passed: boolean, testIdPrefix: string) => {
    if (passed) {
      return (
        <span
          className="chip chip-ok"
          data-testid={`${testIdPrefix}-pass`}
          style={{ fontSize: 'var(--text-xs)' }}
        >
          <CheckCircle2 size={12} style={{ marginRight: '4px' }} /> PASS
        </span>
      );
    }
    return (
      <span
        className="chip chip-bad"
        data-testid={`${testIdPrefix}-fail`}
        style={{ fontSize: 'var(--text-xs)' }}
      >
        <AlertTriangle size={12} style={{ marginRight: '4px' }} /> FAIL
      </span>
    );
  };

  // Column categorization
  const filteredColumns = RAW_COLUMN_NAMES.filter((col) => {
    if (!searchFilter.trim()) return true;
    const q = searchFilter.toLowerCase();
    const sensor = sensorSpecs[col];
    const op = opSettingSpecs[col];
    return (
      col.toLowerCase().includes(q) ||
      (sensor && sensor.name.toLowerCase().includes(q)) ||
      (op && op.description.toLowerCase().includes(q))
    );
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Top: Data Quality Audit Panel */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <ShieldCheck
                size={18}
                style={{
                  color: qualityStats.integrity_status === 'PASS' && qualityStats.structural_status === 'PASS'
                    ? 'var(--color-ok)'
                    : 'var(--color-bad)',
                }}
              />
              <h3 style={{ fontSize: 'var(--text-md)', color: 'var(--color-ink)', margin: 0 }}>
                Data Integrity & Quality Audit ({dataset} {split})
              </h3>
            </div>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Automated structural verification computed directly from committed raw files
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <span
              className={`chip ${qualityStats.structural_status === 'PASS' ? 'chip-ok' : 'chip-bad'}`}
              data-testid="badge-structural-status"
              style={{ fontWeight: 600 }}
            >
              Structural: {qualityStats.structural_status}
            </span>
            <span
              className={`chip ${qualityStats.integrity_status === 'PASS' ? 'chip-ok' : 'chip-bad'}`}
              data-testid="badge-integrity-status"
              style={{ fontWeight: 600 }}
            >
              Integrity: {qualityStats.integrity_status}
            </span>
          </div>
        </div>

        {/* Quality Checks Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: 'var(--space-sm)',
          }}
        >
          {/* Check 1: Canonical Schema Width */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Canonical Schema Width
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>Expected: 26 columns</div>
            </div>
            {renderQualityBadge(qualityStats.column_count_ok && qualityStats.column_names_ok, 'check-schema-width')}
          </div>

          {/* Check 2: Missing & NaN Values */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Missing & NaN Values
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Total missing: <span className="measured">{qualityStats.total_missing_values}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.total_missing_values === 0, 'check-missing-values')}
          </div>

          {/* Check 3: Non-Finite Values */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Non-Finite (Inf / -Inf)
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Total non-finite: <span className="measured">{qualityStats.total_non_finite_values}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.total_non_finite_values === 0, 'check-non-finite')}
          </div>

          {/* Check 4: Duplicate Full Rows */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Duplicate Full Rows
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Identical rows: <span className="measured">{qualityStats.duplicate_rows}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.duplicate_rows === 0, 'check-duplicate-rows')}
          </div>

          {/* Check 5: Duplicate Unit-Cycle Pairs */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Duplicate (Unit, Cycle) Pairs
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Collision count: <span className="measured">{qualityStats.duplicate_unit_cycle}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.duplicate_unit_cycle === 0, 'check-duplicate-unit-cycle')}
          </div>

          {/* Check 6: Cycle Monotonicity */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Cycle Sequence Monotonicity
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Cycles strictly ascending: <span className="measured">{qualityStats.cycles_ordered ? 'Yes' : 'No'}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.cycles_ordered, 'check-cycles-ordered')}
          </div>

          {/* Check 7: Engine Initial Cycle */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Engine Initial Cycle
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                All engines start at cycle 1: <span className="measured">{qualityStats.cycles_start_at_1 ? 'Yes' : 'No'}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.cycles_start_at_1, 'check-cycles-start-at-1')}
          </div>

          {/* Check 8: Unit ID Contiguity */}
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-2)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)' }}>
                Unit ID Contiguity (1..N)
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--color-muted)' }}>
                Contiguous positive integers: <span className="measured">{qualityStats.valid_unit_ids ? 'Yes' : 'No'}</span>
              </div>
            </div>
            {renderQualityBadge(qualityStats.valid_unit_ids, 'check-valid-unit-ids')}
          </div>
        </div>

        {/* Audit Guidance Notes */}
        <div
          style={{
            padding: 'var(--space-sm)',
            background: 'var(--color-paper-3)',
            borderRadius: 'var(--radius-sm)',
            fontSize: 'var(--text-xs)',
            color: 'var(--color-muted)',
            lineHeight: 1.5,
          }}
        >
          <p style={{ margin: 0, marginBottom: '6px' }}>
            <strong>Near-Constant Sensor Rule:</strong> Sensors are evaluated against the repository threshold <code>NEAR_CONSTANT_STD_THRESHOLD = 0.001</code> (where <code>std / |mean| &lt; 0.001</code>). A near-constant sensor is a physical measurement characteristic (such as fan inlet temperature/pressure in single-regime flights), not a data collection failure.
          </p>
          <p style={{ margin: 0 }}>
            <strong>Censoring Semantics:</strong> Test split engines end before failure by design. Shorter test trajectories are authentic operational cutoffs and are not flagged as incomplete observations.
          </p>
        </div>
      </div>

      {/* Bottom: Canonical Schema Dictionary */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h3 style={{ fontSize: 'var(--text-md)', color: 'var(--color-ink)', margin: 0 }}>
              C-MAPSS Canonical 26-Column Variable Dictionary
            </h3>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
              Official space-separated column ordering established in <code>src/config.py</code>
            </span>
          </div>

          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={14} style={{ position: 'absolute', left: '8px', color: 'var(--color-muted)' }} />
            <input
              type="text"
              placeholder="Search variables..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="input-field"
              style={{ paddingLeft: '26px', fontSize: 'var(--text-xs)', width: '180px', height: '30px' }}
              aria-label="Search schema columns"
            />
          </div>
        </div>

        <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
          <table className="table" style={{ width: '100%', fontSize: 'var(--text-xs)', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: 'var(--rule-hair) solid var(--color-rule-2)', textAlign: 'left' }}>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Index</th>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Field Identifier</th>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Category</th>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Physical Name / Description</th>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Unit</th>
                <th style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>Data Type</th>
              </tr>
            </thead>
            <tbody>
              {filteredColumns.map((colKey) => {
                const isId = colKey === 'unit_id' || colKey === 'cycle';
                const isOp = colKey.startsWith('op_setting_');
                const category = isId ? 'Engine Identity' : isOp ? 'Operating Setting' : 'Sensor Measurement';
                const sensor = sensorSpecs[colKey];
                const op = opSettingSpecs[colKey];
                const name = isId
                  ? colKey === 'unit_id'
                    ? 'Unique engine identifier (1..N)'
                    : 'Operational cycle count (1..N)'
                  : isOp
                  ? op?.name || ''
                  : sensor?.name || '';
                const unit = isId ? 'count' : isOp ? op?.unit || '—' : sensor?.unit || '—';
                const dType = isId ? 'int64' : 'float64';

                return (
                  <tr
                    key={colKey}
                    style={{
                      borderBottom: 'var(--rule-hair) solid var(--color-rule)',
                    }}
                  >
                    <td className="measured" style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>
                      #{RAW_COLUMN_NAMES.indexOf(colKey) + 1}
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <span className="measured" style={{ color: 'var(--color-ink)', fontWeight: 600 }}>
                        {colKey}
                      </span>
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      <span
                        className={`chip ${isId ? 'chip-ok' : isOp ? 'chip-warn' : ''}`}
                        style={{ fontSize: '0.65rem' }}
                      >
                        {category}
                      </span>
                    </td>
                    <td style={{ padding: '6px 8px', color: 'var(--color-ink-2)' }}>{name}</td>
                    <td className="measured" style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>{unit}</td>
                    <td className="measured" style={{ padding: '6px 8px', color: 'var(--color-muted)' }}>{dType}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
