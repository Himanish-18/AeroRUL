import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  DatasetId,
  Phase5WindowAblationRecord,
  Phase5TargetCapSensitivityRecord,
  HyperparameterTrialRecord,
  DatasetManifestConfig,
} from '../../lib/data/types';
import {
  getWindowAblationForDataset,
  getTargetCapSensitivityForDataset,
  getHyperparameterTrialsForDataset,
  getBestHyperparameterTrial,
  getHyperparameterTrialSummary,
} from '../../lib/data/modelEvaluationAdapter';
import {
  formatCycles,
  formatRuntime,
  formatNasaScore,
} from '../../lib/utils/format';
import { Layers, Sliders, Sparkles, Eye } from 'lucide-react';

interface ConfigurationTuningSectionProps {
  dataset: DatasetId;
  windowRecords: Phase5WindowAblationRecord[];
  targetCapRecords: Phase5TargetCapSensitivityRecord[];
  hyperparameterTrials: HyperparameterTrialRecord[];
  manifestConfig: DatasetManifestConfig;
}

export const ConfigurationTuningSection: React.FC<ConfigurationTuningSectionProps> = ({
  dataset,
  windowRecords,
  targetCapRecords,
  hyperparameterTrials,
  manifestConfig,
}) => {
  const [modelFamilyFilter, setModelFamilyFilter] = useState<'all' | 'LightGBM' | 'XGBoost'>('LightGBM');
  const [selectedTrialId, setSelectedTrialId] = useState<number | null>(null);

  // Filtered dataset records
  const dsWindowRecords = getWindowAblationForDataset(windowRecords, dataset);
  const dsTargetCapRecords = getTargetCapSensitivityForDataset(targetCapRecords, dataset);
  const dsTrials = getHyperparameterTrialsForDataset(hyperparameterTrials, dataset, modelFamilyFilter);

  const trialSummary = getHyperparameterTrialSummary(dsTrials);
  const bestTrial = getBestHyperparameterTrial(dsTrials);

  // Selected inspected trial (defaults to best trial)
  const inspectedTrial = dsTrials.find((t) => t.trial_id === selectedTrialId) || bestTrial || dsTrials[0];

  // Feature window chart data
  const windowChartData = dsWindowRecords.map((w) => ({
    config: w.window_configuration,
    featureCount: w.feature_count,
    cvMae: Number(w.cv_engine_macro_mae.toFixed(2)),
    valMae: Number(w.val_engine_macro_mae.toFixed(2)),
    isFrozen: w.window_configuration === manifestConfig.feature_configuration,
  }));

  // Target cap chart data (treat 'None' as categorical)
  const targetCapChartData = dsTargetCapRecords.map((c) => ({
    capLabel: c.target_cap === 'None' ? 'None (Unclipped)' : `Cap ${c.target_cap}`,
    cvMae: Number(c.cv_engine_macro_mae.toFixed(2)),
    valMae: Number(c.val_engine_macro_mae.toFixed(2)),
    isFrozen: String(c.target_cap) === String(manifestConfig.target_cap),
  }));

  // Trial history chart data
  const trialChartData = dsTrials.map((t) => ({
    trial_id: t.trial_id,
    mae: Number(t.engine_macro_mae.toFixed(2)),
    model_family: t.model_family,
    isBest: t.trial_id === bestTrial?.trial_id,
    runtime: t.runtime_sec,
  }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* 1. Feature Window Ablation Study */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <Layers size={18} color="var(--color-accent)" />
              <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
                Feature-Window Ablation Study: {dataset}
              </h2>
            </div>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Assessing temporal window horizon tradeoffs. Frozen feature configuration: <strong>{manifestConfig.feature_configuration}</strong> ({manifestConfig.n_features} features).
            </p>
          </div>
          <span className="chip chip-accent">
            frozen: {manifestConfig.feature_configuration}
          </span>
        </div>

        {dataset === 'FD002' && (
          <div
            style={{
              padding: 'var(--space-xs) var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule-2)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-2)',
            }}
          >
            <strong>FD002 Feature Selection Rationale:</strong> W75 (231 features) was selected over ALL_WINDOWS (670 features) because it achieved optimal CV macro MAE ({formatCycles(manifestConfig.inner_cv_scores.engine_macro_mae, 2)} cycles) with 65.5% fewer features and superior generalization.
          </div>
        )}

        <div style={{ height: '220px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={windowChartData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid stroke="var(--color-rule)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="config" stroke="var(--color-ink-2)" fontSize={11} />
              <YAxis stroke="var(--color-muted)" fontSize={11} unit=" cyc" domain={[0, 'auto']} />
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
                      <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>
                        {item.config} ({item.featureCount} features)
                      </div>
                      <div style={{ color: 'var(--color-accent)' }}>CV Macro MAE: {item.cvMae} cyc</div>
                      <div style={{ color: 'var(--color-cool)' }}>Dev-Val Macro MAE: {item.valMae} cyc</div>
                      {item.isFrozen && (
                        <div style={{ color: 'var(--color-accent)', fontWeight: 600, marginTop: '2px' }}>
                          ✓ Frozen Selection
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '11px', paddingBottom: '4px' }} />
              <Bar name="CV Macro MAE" dataKey="cvMae" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
              <Bar name="Dev-Val Macro MAE" dataKey="valMae" fill="var(--color-cool)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-container">
          <table className="data-table" aria-label="Feature Window Ablation Metrics">
            <thead>
              <tr>
                <th>Window Configuration</th>
                <th style={{ textAlign: 'center' }}>Target Window</th>
                <th style={{ textAlign: 'right' }}>Feature Count</th>
                <th style={{ textAlign: 'right' }}>CV Macro MAE (cyc)</th>
                <th style={{ textAlign: 'right' }}>CV Pooled MAE</th>
                <th style={{ textAlign: 'right' }}>Dev-Val Macro MAE</th>
                <th style={{ textAlign: 'right' }}>Dev-Val NASA Score</th>
                <th style={{ textAlign: 'center' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {dsWindowRecords.map((row) => {
                const isFrozen = row.window_configuration === manifestConfig.feature_configuration;
                return (
                  <tr key={row.window_configuration} className={isFrozen ? 'selected' : ''}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                        <span style={{ fontWeight: isFrozen ? 600 : 500, color: isFrozen ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
                          {row.window_configuration}
                        </span>
                        {isFrozen && (
                          <span className="chip chip-accent" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            frozen
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={{ textAlign: 'center' }}>{row.target_window}</td>
                    <td className="measured" style={{ textAlign: 'right' }}>{row.feature_count}</td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: 600, color: isFrozen ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.cv_engine_macro_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>{formatCycles(row.cv_pooled_mae, 2)}</td>
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-cool)', fontWeight: 500 }}>
                      {formatCycles(row.val_engine_macro_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>{formatNasaScore(row.val_nasa_score_mean)}</td>
                    <td style={{ textAlign: 'center' }}>
                      {isFrozen ? (
                        <span className="chip chip-accent" style={{ fontSize: '0.65rem' }}>selected</span>
                      ) : (
                        <span className="chip" style={{ fontSize: '0.65rem' }}>ablated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Target-Cap Sensitivity Study */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <Sliders size={18} color="var(--color-accent)" />
              <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
                Target-Cap Sensitivity Analysis: {dataset}
              </h2>
            </div>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Examining piecewise upper clipping bounds. Frozen target cap: <strong>{manifestConfig.target_cap}</strong> ({manifestConfig.output_constraint}).
            </p>
          </div>
          <span className="chip chip-accent">
            frozen cap: {manifestConfig.target_cap}
          </span>
        </div>

        <div style={{ height: '220px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={targetCapChartData} margin={{ top: 10, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid stroke="var(--color-rule)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="capLabel" stroke="var(--color-ink-2)" fontSize={11} />
              <YAxis stroke="var(--color-muted)" fontSize={11} unit=" cyc" domain={[0, 'auto']} />
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
                      <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>{item.capLabel}</div>
                      <div style={{ color: 'var(--color-accent)' }}>CV Macro MAE: {item.cvMae} cyc</div>
                      <div style={{ color: 'var(--color-cool)' }}>Dev-Val Macro MAE: {item.valMae} cyc</div>
                      {item.isFrozen && (
                        <div style={{ color: 'var(--color-accent)', fontWeight: 600, marginTop: '2px' }}>
                          ✓ Frozen Selection
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: '11px', paddingBottom: '4px' }} />
              <Bar name="CV Macro MAE" dataKey="cvMae" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
              <Bar name="Dev-Val Macro MAE" dataKey="valMae" fill="var(--color-cool)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="table-container">
          <table className="data-table" aria-label="Target Cap Sensitivity Metrics">
            <thead>
              <tr>
                <th>Target Cap Bound</th>
                <th>Output Constraint Formula</th>
                <th style={{ textAlign: 'right' }}>CV Macro MAE (cyc)</th>
                <th style={{ textAlign: 'right' }}>CV Pooled MAE</th>
                <th style={{ textAlign: 'right' }}>Dev-Val Macro MAE</th>
                <th style={{ textAlign: 'right' }}>Dev-Val NASA Score</th>
                <th style={{ textAlign: 'right' }}>Validation Engines</th>
                <th style={{ textAlign: 'center' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {dsTargetCapRecords.map((row) => {
                const isFrozen = String(row.target_cap) === String(manifestConfig.target_cap);
                return (
                  <tr key={String(row.target_cap)} className={isFrozen ? 'selected' : ''}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                        <span style={{ fontWeight: isFrozen ? 600 : 500, color: isFrozen ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
                          {row.target_cap === 'None' ? 'None (Unclipped)' : `Cap ${row.target_cap}`}
                        </span>
                        {isFrozen && (
                          <span className="chip chip-accent" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            frozen
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <code style={{ fontSize: '0.75rem', color: 'var(--color-ink-2)' }}>
                        {row.output_constraint}
                      </code>
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: 600, color: isFrozen ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.cv_engine_macro_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>{formatCycles(row.cv_pooled_mae, 2)}</td>
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-cool)', fontWeight: 500 }}>
                      {formatCycles(row.val_engine_macro_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>{formatNasaScore(row.val_nasa_score_mean)}</td>
                    <td className="measured" style={{ textAlign: 'right' }}>{row.n_val_engines}</td>
                    <td style={{ textAlign: 'center' }}>
                      {isFrozen ? (
                        <span className="chip chip-accent" style={{ fontSize: '0.65rem' }}>selected</span>
                      ) : (
                        <span className="chip" style={{ fontSize: '0.65rem' }}>evaluated</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Optuna Hyperparameter Search Explorer */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <Sparkles size={18} color="var(--color-accent)" />
              <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
                Optuna Hyperparameter Search Explorer: {dataset}
              </h2>
            </div>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Interactive inspection of stored TPE trials. Click any trial point or row to inspect tuned hyperparameters.
            </p>
          </div>

          {/* Model Family Filter Controls */}
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
            {(['LightGBM', 'XGBoost', 'all'] as const).map((fam) => (
              <button
                key={fam}
                type="button"
                onClick={() => setModelFamilyFilter(fam)}
                className="btn btn-sm"
                style={{
                  background: modelFamilyFilter === fam ? 'var(--color-paper-2)' : 'transparent',
                  borderColor: modelFamilyFilter === fam ? 'var(--color-rule-2)' : 'transparent',
                  color: modelFamilyFilter === fam ? 'var(--color-ink)' : 'var(--color-ink-2)',
                  fontWeight: modelFamilyFilter === fam ? 600 : 400,
                  fontSize: '0.75rem',
                  padding: '0 var(--space-sm)',
                  height: '26px',
                }}
              >
                {fam === 'all' ? 'All Families' : fam}
              </button>
            ))}
          </div>
        </div>

        {/* Trial Discrepancy Notice */}
        <div
          style={{
            padding: 'var(--space-xs) var(--space-sm)',
            background: 'var(--color-paper-3)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule-2)',
            fontSize: 'var(--text-xs)',
            color: 'var(--color-muted)',
            lineHeight: 1.45,
          }}
        >
          <strong>Artifact Provenance Notice:</strong> The committed hyperparameter search CSV contains{' '}
          <strong>15 LightGBM trials</strong> and <strong>10 XGBoost trials</strong> per dataset (25 trials per dataset; 100 total). This dashboard strictly displays stored trial records without fabricating missing runs.
        </div>

        {/* Trial Summary Metrics Grid */}
        {trialSummary && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 'var(--space-sm)' }}>
            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Stored Trials</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {trialSummary.completedTrials} / {trialSummary.totalStoredTrials} completed
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Best Engine-Macro MAE</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-accent)' }}>
                {formatCycles(trialSummary.bestEngineMacroMae, 2)} cyc
              </div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
                trial #{trialSummary.bestTrialId}
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Median Macro MAE</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {formatCycles(trialSummary.medianEngineMacroMae, 2)} cyc
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Best Pooled MAE / RMSE</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {formatCycles(trialSummary.bestPooledMae, 2)} / {formatCycles(trialSummary.bestPooledRmse, 2)} cyc
              </div>
            </div>

            <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Best Trial Runtime</div>
              <div style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)' }}>
                {formatRuntime(trialSummary.runtimeSecBestTrial)}
              </div>
            </div>
          </div>
        )}

        {/* Trial History Chart */}
        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={trialChartData}
              margin={{ top: 10, right: 30, left: 10, bottom: 5 }}
              onClick={(e) => {
                if (e && e.activePayload && e.activePayload.length > 0) {
                  setSelectedTrialId(e.activePayload[0].payload.trial_id);
                }
              }}
            >
              <CartesianGrid stroke="var(--color-rule)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="trial_id" stroke="var(--color-ink-2)" fontSize={11} label={{ value: 'Trial ID', position: 'insideBottomRight', offset: -5, fontSize: 10, fill: 'var(--color-muted)' }} />
              <YAxis stroke="var(--color-muted)" fontSize={11} unit=" cyc" domain={['auto', 'auto']} />
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
                      <div style={{ fontWeight: 600, color: 'var(--color-ink)' }}>
                        Trial #{item.trial_id} ({item.model_family})
                      </div>
                      <div style={{ color: 'var(--color-accent)' }}>Macro MAE: {item.mae} cycles</div>
                      <div style={{ color: 'var(--color-muted)' }}>Runtime: {formatRuntime(item.runtime)}</div>
                      {item.isBest && (
                        <div style={{ color: 'var(--color-accent)', fontWeight: 600, marginTop: '2px' }}>
                          ★ Best Stored Trial
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Line
                type="monotone"
                dataKey="mae"
                stroke="var(--color-accent)"
                strokeWidth={2}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  const isSelected = payload.trial_id === inspectedTrial?.trial_id;
                  const isBest = payload.isBest;
                  return (
                    <circle
                      key={`dot-${payload.trial_id}`}
                      cx={cx}
                      cy={cy}
                      r={isSelected ? 6 : isBest ? 5 : 3.5}
                      fill={isBest ? 'var(--color-accent)' : isSelected ? 'var(--color-ink)' : 'var(--color-paper-3)'}
                      stroke={isSelected ? 'var(--color-accent)' : 'var(--color-rule-2)'}
                      strokeWidth={isSelected ? 2 : 1}
                      style={{ cursor: 'pointer' }}
                    />
                  );
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Trial Detail Inspector Card */}
        {inspectedTrial && (
          <div
            style={{
              padding: 'var(--space-md)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-card)',
              border: 'var(--rule-hair) solid var(--color-rule-2)',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-sm)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                <Eye size={16} color="var(--color-accent)" />
                <span style={{ fontWeight: 600, fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
                  Inspecting Trial #{inspectedTrial.trial_id}: {inspectedTrial.model_family} ({dataset})
                </span>
                {inspectedTrial.trial_id === bestTrial?.trial_id && (
                  <span className="chip chip-accent" style={{ fontSize: '0.65rem' }}>
                    best trial
                  </span>
                )}
                <span className="chip" style={{ fontSize: '0.65rem' }}>
                  {inspectedTrial.status}
                </span>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-md)', fontSize: 'var(--text-xs)' }}>
                <span>
                  <strong style={{ color: 'var(--color-muted)' }}>Macro MAE:</strong>{' '}
                  <span className="measured" style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                    {formatCycles(inspectedTrial.engine_macro_mae, 4)} cyc
                  </span>
                </span>
                <span>
                  <strong style={{ color: 'var(--color-muted)' }}>Pooled MAE / RMSE:</strong>{' '}
                  <span className="measured">
                    {formatCycles(inspectedTrial.pooled_mae, 2)} / {formatCycles(inspectedTrial.pooled_rmse, 2)}
                  </span>
                </span>
                <span>
                  <strong style={{ color: 'var(--color-muted)' }}>Runtime:</strong>{' '}
                  <span className="measured">{formatRuntime(inspectedTrial.runtime_sec)}</span>
                </span>
              </div>
            </div>

            {/* Parsed JSON Parameters Grid */}
            <div>
              <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-muted)', marginBottom: 'var(--space-2xs)' }}>
                Trial Hyperparameter JSON Parameters
              </div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))',
                  gap: 'var(--space-2xs)',
                  padding: 'var(--space-sm)',
                  background: 'var(--color-paper-2)',
                  borderRadius: 'var(--radius-sm)',
                  border: 'var(--rule-hair) solid var(--color-rule)',
                  fontFamily: 'var(--font-outlier)',
                  fontSize: 'var(--text-xs)',
                }}
              >
                {Object.entries(inspectedTrial.params).map(([key, val]) => (
                  <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--space-xs)' }}>
                    <span style={{ color: 'var(--color-muted)' }}>{key}:</span>
                    <span style={{ color: 'var(--color-ink)', fontWeight: 500 }}>
                      {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
