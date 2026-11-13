import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import {
  DatasetId,
  Phase4BenchmarkRecord,
  Phase4SortMetric,
} from '../../lib/data/types';
import {
  sortPhase4Candidates,
  getPhase4Leader,
} from '../../lib/data/modelEvaluationAdapter';
import {
  formatCycles,
  formatR2,
  formatNasaScore,
  formatRuntime,
  formatSignedError,
} from '../../lib/utils/format';
import { Cpu } from 'lucide-react';

interface Phase4BenchmarkSectionProps {
  dataset: DatasetId;
  records: Phase4BenchmarkRecord[];
}

export const Phase4BenchmarkSection: React.FC<Phase4BenchmarkSectionProps> = ({
  dataset,
  records,
}) => {
  const [sortMetric, setSortMetric] = useState<Phase4SortMetric>('endpoint_mae');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const sortedRecords = sortPhase4Candidates(records, sortMetric, sortDirection);
  const leaderEndpoint = getPhase4Leader(records, 'endpoint_mae');
  const leaderRow = getPhase4Leader(records, 'row_mae');
  const activeLeader = getPhase4Leader(records, sortMetric);

  // Active inspected model (defaults to active metric leader or endpoint leader)
  const inspectedRecord = records.find((r) => r.model === selectedModel) || activeLeader || leaderEndpoint || records[0];

  const validationEngines = records.length > 0 ? records[0].n_engines : 0;

  const handleMetricToggle = (metric: Phase4SortMetric) => {
    if (sortMetric === metric) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortMetric(metric);
      setSortDirection('asc');
    }
  };

  const metricLabels: Record<Phase4SortMetric, string> = {
    endpoint_mae: 'Endpoint MAE',
    row_mae: 'Row MAE',
    macro_mae: 'Engine-Macro MAE',
    row_rmse: 'Row RMSE',
  };

  // Chart data sorted by active metric
  const chartData = [...records].sort((a, b) => a[sortMetric] - b[sortMetric]).map((r) => ({
    model: r.model,
    value: Number(r[sortMetric].toFixed(2)),
    isLeader: r.model === activeLeader?.model,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* 1. Summary Cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 'var(--space-md)',
        }}
      >
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3xs)' }}>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Evaluated Candidates</div>
          <div style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-ink)' }}>
            {records.length} models
          </div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
            classical ML baseline pool
          </div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3xs)' }}>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Best Endpoint MAE</div>
          <div style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-accent)' }}>
            {leaderEndpoint ? `${formatCycles(leaderEndpoint.endpoint_mae, 2)} cyc` : '—'}
          </div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
            leader: {leaderEndpoint?.model || '—'}
          </div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3xs)' }}>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Best Row MAE</div>
          <div style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-ink)' }}>
            {leaderRow ? `${formatCycles(leaderRow.row_mae, 2)} cyc` : '—'}
          </div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
            leader: {leaderRow?.model || '—'}
          </div>
        </div>

        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3xs)' }}>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Validation Cohort</div>
          <div style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-ink)' }}>
            {validationEngines} engines
          </div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)' }}>
            80/20 train-validation split (seed 42)
          </div>
        </div>
      </div>

      {/* 2. Interactive Comparison Chart with Metric Selector */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Phase 4 Candidate Model Comparison: {dataset}
            </h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Visualizing {metricLabels[sortMetric]}. Lower is better. Tap any bar or row to inspect configuration details.
            </p>
          </div>

          {/* Metric Selector Tabs */}
          <div
            style={{
              display: 'inline-flex',
              padding: '2px',
              background: 'var(--color-paper-3)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              borderRadius: 'var(--radius-pill)',
              gap: '2px',
            }}
          >
            {(['endpoint_mae', 'row_mae', 'macro_mae', 'row_rmse'] as Phase4SortMetric[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setSortMetric(m)}
                className="btn btn-sm"
                style={{
                  background: sortMetric === m ? 'var(--color-paper-2)' : 'transparent',
                  borderColor: sortMetric === m ? 'var(--color-rule-2)' : 'transparent',
                  color: sortMetric === m ? 'var(--color-ink)' : 'var(--color-ink-2)',
                  fontWeight: sortMetric === m ? 600 : 400,
                  fontSize: '0.75rem',
                  padding: '0 var(--space-xs)',
                  height: '26px',
                }}
              >
                {metricLabels[m]}
              </button>
            ))}
          </div>
        </div>

        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 110, bottom: 5 }}
            >
              <CartesianGrid stroke="var(--color-rule)" strokeDasharray="3 3" horizontal={true} vertical={false} />
              <XAxis
                type="number"
                stroke="var(--color-muted)"
                fontSize={11}
                unit=" cyc"
                domain={[0, 'auto']}
              />
              <YAxis
                type="category"
                dataKey="model"
                stroke="var(--color-ink-2)"
                fontSize={11}
                tickLine={false}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload || payload.length === 0) return null;
                  const item = payload[0].payload;
                  return (
                    <div
                      style={{
                        background: 'var(--color-paper-3)',
                        border: 'var(--rule-hair) solid var(--color-rule-2)',
                        padding: 'var(--space-xs) var(--space-sm)',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: 'var(--text-xs)',
                      }}
                    >
                      <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>{item.model}</div>
                      <div style={{ color: item.isLeader ? 'var(--color-accent)' : 'var(--color-ink-2)' }}>
                        {metricLabels[sortMetric]}: {item.value} cycles
                      </div>
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="value"
                radius={[0, 4, 4, 0]}
                onClick={(entry) => setSelectedModel(entry.model)}
                style={{ cursor: 'pointer' }}
              >
                {chartData.map((entry) => (
                  <Cell
                    key={`cell-${entry.model}`}
                    fill={entry.model === inspectedRecord?.model ? 'var(--color-accent)' : 'var(--color-paper-3)'}
                    stroke={entry.model === inspectedRecord?.model ? 'var(--color-accent)' : 'var(--color-rule-2)'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Candidate Leaderboard Table */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Phase 4 Candidate Leaderboard (Held-Out Validation Benchmark)
            </h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Default ranked by Endpoint MAE. Click column header to change sort order. Endpoint R² is excluded from ranking as actual endpoint RULs are zero.
            </p>
          </div>
          <span className="chip" style={{ fontSize: '0.7rem' }}>
            6 candidates evaluated
          </span>
        </div>

        <div className="table-container">
          <table className="data-table" aria-label="Phase 4 Benchmark Leaderboard">
            <thead>
              <tr>
                <th style={{ width: '40px', textAlign: 'center' }}>#</th>
                <th>Candidate Model</th>
                <th style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleMetricToggle('endpoint_mae')}
                    style={{ fontWeight: sortMetric === 'endpoint_mae' ? 600 : 400 }}
                  >
                    Endpoint MAE (cyc) {sortMetric === 'endpoint_mae' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </button>
                </th>
                <th style={{ textAlign: 'right' }}>Endpoint RMSE</th>
                <th style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleMetricToggle('row_mae')}
                    style={{ fontWeight: sortMetric === 'row_mae' ? 600 : 400 }}
                  >
                    Row MAE (cyc) {sortMetric === 'row_mae' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </button>
                </th>
                <th style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleMetricToggle('row_rmse')}
                    style={{ fontWeight: sortMetric === 'row_rmse' ? 600 : 400 }}
                  >
                    Row RMSE {sortMetric === 'row_rmse' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </button>
                </th>
                <th style={{ textAlign: 'right' }}>Row R²</th>
                <th style={{ textAlign: 'right' }}>
                  <button
                    type="button"
                    className="table-sort-btn"
                    onClick={() => handleMetricToggle('macro_mae')}
                    style={{ fontWeight: sortMetric === 'macro_mae' ? 600 : 400 }}
                  >
                    Macro MAE {sortMetric === 'macro_mae' ? (sortDirection === 'asc' ? '↑' : '↓') : ''}
                  </button>
                </th>
                <th style={{ textAlign: 'right' }}>NASA Score</th>
                <th style={{ textAlign: 'right' }}>Runtime</th>
                <th style={{ textAlign: 'center' }}>Inspect</th>
              </tr>
            </thead>
            <tbody>
              {sortedRecords.map((row, idx) => {
                const isInspected = row.model === inspectedRecord?.model;
                const isLeader = row.model === activeLeader?.model;

                return (
                  <tr
                    key={row.model}
                    className={isInspected ? 'selected' : ''}
                    onClick={() => setSelectedModel(row.model)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td style={{ textAlign: 'center', fontWeight: 600, color: 'var(--color-muted)' }}>
                      #{idx + 1}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                        <span style={{ fontWeight: isLeader ? 600 : 500, color: isLeader ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
                          {row.model}
                        </span>
                        {isLeader && (
                          <span className="chip chip-accent" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            leader
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: sortMetric === 'endpoint_mae' && isLeader ? 600 : 400, color: sortMetric === 'endpoint_mae' && isLeader ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.endpoint_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.endpoint_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: sortMetric === 'row_mae' && isLeader ? 600 : 400, color: sortMetric === 'row_mae' && isLeader ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.row_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: sortMetric === 'row_rmse' && isLeader ? 600 : 400, color: sortMetric === 'row_rmse' && isLeader ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.row_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatR2(row.row_r2, 3)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: sortMetric === 'macro_mae' && isLeader ? 600 : 400, color: sortMetric === 'macro_mae' && isLeader ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.macro_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatNasaScore(row.endpoint_nasa_score)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-muted)' }}>
                      {formatRuntime(row.runtime)}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <button
                        type="button"
                        className="btn btn-sm"
                        style={{
                          height: '24px',
                          padding: '0 var(--space-xs)',
                          fontSize: '0.65rem',
                          background: isInspected ? 'var(--color-accent)' : 'transparent',
                          color: isInspected ? 'var(--color-accent-ink)' : 'var(--color-ink-2)',
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedModel(row.model);
                        }}
                      >
                        {isInspected ? 'active' : 'select'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Model Detail Panel (Inspected Candidate) */}
      {inspectedRecord && (
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <Cpu size={18} color="var(--color-accent)" />
              <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
                Candidate Detail: {inspectedRecord.model} ({dataset})
              </h2>
            </div>
            <span className="chip">
              Phase 4 Fixed Starting Specification
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-sm)' }}>
            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Endpoint MAE / RMSE</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-accent)' }}>
                {formatCycles(inspectedRecord.endpoint_mae, 2)} / {formatCycles(inspectedRecord.endpoint_rmse, 2)} cyc
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Row MAE / Row R²</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {formatCycles(inspectedRecord.row_mae, 2)} cyc / {formatR2(inspectedRecord.row_r2, 3)}
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Mean Signed Error / NASA Score</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {formatSignedError(inspectedRecord.mean_signed_error, 2)} cyc / {formatNasaScore(inspectedRecord.endpoint_nasa_score)}
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Validation Cohort / Runtime</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {inspectedRecord.n_engines} engines / {formatRuntime(inspectedRecord.runtime)}
              </div>
            </div>
          </div>

          <div
            style={{
              padding: 'var(--space-xs) var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-muted)',
              lineHeight: 1.45,
            }}
          >
            <strong>Phase 4 Protocol Notice:</strong> Models evaluated with Phase 3.1 features under a 75-cycle window limit. Linear models used StandardScaler fitted on training splits. Endpoint actual RUL values are zero in the terminal validation endpoint artifact. Not official test performance.
          </div>
        </div>
      )}
    </div>
  );
};
