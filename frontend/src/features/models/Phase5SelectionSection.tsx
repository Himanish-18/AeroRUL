import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  DatasetId,
  Phase5CrossValidationRecord,
  Phase5DevelopmentValidationRecord,
  DatasetManifestConfig,
} from '../../lib/data/types';
import {
  sortPhase5CvCandidates,
  sortPhase5DevValCandidates,
  compareCvAndDevValRanks,
  calculateCvSelectionMargin,
} from '../../lib/data/modelEvaluationAdapter';
import {
  formatCycles,
  formatSignedError,
  formatNasaScore,
} from '../../lib/utils/format';
import { ShieldCheck, AlertCircle, FileCode } from 'lucide-react';

interface Phase5SelectionSectionProps {
  dataset: DatasetId;
  cvRecords: Phase5CrossValidationRecord[];
  devValRecords: Phase5DevelopmentValidationRecord[];
  manifestConfig: DatasetManifestConfig;
}

export const Phase5SelectionSection: React.FC<Phase5SelectionSectionProps> = ({
  dataset,
  cvRecords,
  devValRecords,
  manifestConfig,
}) => {
  const sortedCv = sortPhase5CvCandidates(cvRecords, 'asc');
  const sortedDevVal = sortPhase5DevValCandidates(devValRecords, 'asc');
  const rankComparison = compareCvAndDevValRanks(cvRecords, devValRecords);
  const selectionMargin = calculateCvSelectionMargin(cvRecords, manifestConfig.model_family);

  // Prepare chart data comparing CV vs Dev Val for matching candidates
  const chartData = sortedCv.map((cv) => {
    const dev = devValRecords.find((d) => d.model === cv.model);
    return {
      model: cv.model.replace('_Tuned', ' (Tuned)').replace('_Fixed', ' (Fixed)'),
      modelRaw: cv.model,
      cvMae: Number(cv.engine_macro_mae.toFixed(2)),
      devMae: dev ? Number(dev.engine_macro_mae.toFixed(2)) : null,
      isFrozen: cv.model.toLowerCase().includes(manifestConfig.model_family.toLowerCase()),
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* 1. Selection Margin Callout Banner */}
      {selectionMargin && (
        <div
          style={{
            background: selectionMargin.isNarrow ? 'var(--color-paper-2)' : 'var(--color-paper-2)',
            border: `var(--rule-hair) solid ${selectionMargin.isNarrow ? 'var(--color-accent)' : 'var(--color-rule-2)'}`,
            borderRadius: 'var(--radius-card)',
            padding: 'var(--space-md)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-2xs)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-sm)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <ShieldCheck size={18} color="var(--color-accent)" />
              <span style={{ fontWeight: 600, fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
                Frozen Model Selection: {manifestConfig.model_family} ({manifestConfig.feature_configuration})
              </span>
              <span className="chip chip-accent">
                inner CV winner
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>selection margin:</span>
              <span className="chip" style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                {formatCycles(selectionMargin.absoluteMargin, 4)} cycles ({selectionMargin.percentageMargin.toFixed(2)}%)
              </span>
            </div>
          </div>

          <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--color-ink-2)', lineHeight: 1.45 }}>
            {dataset === 'FD004' ? (
              <span>
                <strong>FD004 Close Competition Notice:</strong> LightGBM won inner CV with an engine-macro MAE of{' '}
                <strong>{formatCycles(selectionMargin.selectedMae, 4)} cycles</strong> versus XGBoost at{' '}
                <strong>{formatCycles(selectionMargin.runnerUpMae, 4)} cycles</strong> (a razor-thin margin of{' '}
                <strong>{formatCycles(selectionMargin.absoluteMargin, 4)} cycles</strong>). LightGBM was selected based strictly on the a priori declared cross-validation criterion.
              </span>
            ) : (
              <span>
                LightGBM ({formatCycles(selectionMargin.selectedMae, 2)} cycles) led the runner-up candidate{' '}
                {selectionMargin.runnerUpModel} ({formatCycles(selectionMargin.runnerUpMae, 2)} cycles) by a margin of{' '}
                {formatCycles(selectionMargin.absoluteMargin, 2)} cycles on engine-macro MAE.
              </span>
            )}
          </p>
        </div>
      )}

      {/* 2. Inner-CV Leaderboard */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Inner Cross-Validation Leaderboard (Primary Selection Protocol)
            </h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              3-fold engine-grouped GroupKFold with positive-RUL simulated checkpoints. Primary selection metric is engine-macro MAE.
            </p>
          </div>
          <span className="chip" style={{ fontSize: '0.7rem' }}>
            criterion: inner_cv_engine_macro_mae
          </span>
        </div>

        <div className="table-container">
          <table className="data-table" aria-label="Phase 5 Inner Cross-Validation Leaderboard">
            <thead>
              <tr>
                <th style={{ width: '40px', textAlign: 'center' }}>Rank</th>
                <th>Candidate Model</th>
                <th style={{ textAlign: 'right' }}>Engine-Macro MAE (cycles)</th>
                <th style={{ textAlign: 'right' }}>± Std (variability)</th>
                <th style={{ textAlign: 'right' }}>Macro RMSE</th>
                <th style={{ textAlign: 'right' }}>Pooled MAE</th>
                <th style={{ textAlign: 'right' }}>Pooled RMSE</th>
                <th style={{ textAlign: 'right' }}>Signed Error</th>
                <th style={{ textAlign: 'right' }}>Mean NASA Score</th>
                <th style={{ textAlign: 'center' }}>Selection Status</th>
              </tr>
            </thead>
            <tbody>
              {sortedCv.map((row, idx) => {
                const isSelected = row.model.toLowerCase().includes(manifestConfig.model_family.toLowerCase());
                return (
                  <tr key={row.model} className={isSelected ? 'selected' : ''}>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: 'var(--color-muted)' }}>
                      #{idx + 1}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                        <span style={{ fontWeight: isSelected ? 600 : 500, color: isSelected ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
                          {row.model}
                        </span>
                        {isSelected && (
                          <span className="chip chip-accent" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            selected
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: 600, color: isSelected ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.engine_macro_mae, 4)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right', color: 'var(--color-muted)' }}>
                      ±{formatCycles(row.engine_macro_mae_std, 4)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.engine_macro_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.pooled_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.pooled_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatSignedError(row.mean_signed_error, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatNasaScore(row.nasa_score_mean)}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {isSelected ? (
                        <span className="chip chip-accent" style={{ fontSize: '0.65rem' }}>
                          frozen choice
                        </span>
                      ) : (
                        <span className="chip" style={{ fontSize: '0.65rem' }}>
                          evaluated
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. Comparison Chart: CV vs Dev Validation */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div>
          <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
            Cross-Validation versus Development Validation (Engine-Macro MAE)
          </h2>
          <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
            Side-by-side comparison on identical scale. Lower is better. These represent two distinct evaluation partitions, not pooled observations.
          </p>
        </div>

        <div style={{ height: '240px', width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 100, bottom: 5 }}
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
                      <div style={{ color: 'var(--color-accent)' }}>
                        Phase 5 Inner CV: {item.cvMae} cycles
                      </div>
                      {item.devMae !== null && (
                        <div style={{ color: 'var(--color-cool)' }}>
                          Development Validation: {item.devMae} cycles
                        </div>
                      )}
                    </div>
                  );
                }}
              />
              <Legend
                verticalAlign="top"
                align="right"
                wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }}
              />
              <Bar
                name="Phase 5 Inner CV"
                dataKey="cvMae"
                fill="var(--color-accent)"
                radius={[0, 4, 4, 0]}
              />
              <Bar
                name="Development Validation"
                dataKey="devMae"
                fill="var(--color-cool)"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 4. Development Validation Table & Reversal Analysis */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div>
            <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Development Validation Partition Evaluation
            </h2>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0 }}>
              Evaluated on independent held-out development validation engines. Shows rank consistency against inner-CV results.
            </p>
          </div>
          <span className="chip" style={{ fontSize: '0.7rem' }}>
            held-out validation partition
          </span>
        </div>

        {/* Reversal / Rank Consistency Note */}
        {dataset === 'FD004' && (
          <div
            style={{
              padding: 'var(--space-xs) var(--space-sm)',
              background: 'var(--color-paper-3)',
              border: 'var(--rule-hair) solid var(--color-rule-2)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-2)',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-xs)',
            }}
          >
            <AlertCircle size={16} color="var(--color-accent)" style={{ flexShrink: 0 }} />
            <span>
              <strong>FD004 Rank Reversal:</strong> On the development validation partition, XGBoost_Tuned narrowly outperformed LightGBM_Tuned (27.5943 vs 27.6324 cycles, Δ = 0.0381 cycles). The frozen decision adheres to the pre-registered inner-CV criterion.
            </span>
          </div>
        )}

        {dataset === 'FD002' && (
          <div
            style={{
              padding: 'var(--space-xs) var(--space-sm)',
              background: 'var(--color-paper-3)',
              border: 'var(--rule-hair) solid var(--color-rule-2)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-ink-2)',
              display: 'flex',
              alignItems: 'center',
              gap: 'var(--space-xs)',
            }}
          >
            <AlertCircle size={16} color="var(--color-cool)" style={{ flexShrink: 0 }} />
            <span>
              <strong>FD002 Target-Cap Challenger:</strong> <code style={{ color: 'var(--color-cool)' }}>LightGBM_Tuned_Cap125</code> (15.24 cycles) is a target-cap sensitivity variant, not a separate model family. The baseline LightGBM_Tuned (uncapped in this row) scored 16.65 cycles.
            </span>
          </div>
        )}

        <div className="table-container">
          <table className="data-table" aria-label="Development Validation Metrics">
            <thead>
              <tr>
                <th style={{ width: '40px', textAlign: 'center' }}>Rank</th>
                <th>Model / Configuration Variant</th>
                <th style={{ textAlign: 'center' }}>Target Cap</th>
                <th style={{ textAlign: 'right' }}>Engine-Macro MAE (cycles)</th>
                <th style={{ textAlign: 'right' }}>Macro RMSE</th>
                <th style={{ textAlign: 'right' }}>Pooled MAE</th>
                <th style={{ textAlign: 'right' }}>Pooled RMSE</th>
                <th style={{ textAlign: 'right' }}>Signed Error</th>
                <th style={{ textAlign: 'right' }}>Mean NASA Score</th>
                <th style={{ textAlign: 'right' }}>Engines (n)</th>
                <th style={{ textAlign: 'center' }}>CV vs Dev Rank</th>
              </tr>
            </thead>
            <tbody>
              {sortedDevVal.map((row, idx) => {
                const isSelected = row.model.toLowerCase().includes(manifestConfig.model_family.toLowerCase()) && !row.model.includes('Cap125');
                const comp = rankComparison.find((c) => c.model === row.model);

                return (
                  <tr key={row.model} className={isSelected ? 'selected' : ''}>
                    <td style={{ textAlign: 'center', fontWeight: 600, color: 'var(--color-muted)' }}>
                      #{idx + 1}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
                        <span style={{ fontWeight: isSelected ? 600 : 500, color: isSelected ? 'var(--color-ink)' : 'var(--color-ink-2)' }}>
                          {row.model}
                        </span>
                        {isSelected && (
                          <span className="chip chip-accent" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            selected
                          </span>
                        )}
                        {row.model.includes('Cap125') && (
                          <span className="chip" style={{ fontSize: '0.65rem', padding: '0 4px' }}>
                            challenger
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <span className="chip" style={{ fontSize: '0.65rem' }}>
                        {row.target_cap === 'None' ? 'None' : `${row.target_cap}`}
                      </span>
                    </td>
                    <td className="measured" style={{ textAlign: 'right', fontWeight: 600, color: isSelected ? 'var(--color-accent)' : 'var(--color-ink)' }}>
                      {formatCycles(row.engine_macro_mae, 4)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.engine_macro_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.pooled_mae, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatCycles(row.pooled_rmse, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatSignedError(row.mean_signed_error, 2)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {formatNasaScore(row.nasa_score_mean)}
                    </td>
                    <td className="measured" style={{ textAlign: 'right' }}>
                      {row.n_unique_engines}
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {comp ? (
                        <span
                          className={`chip ${comp.isReversal ? 'chip-accent' : ''}`}
                          style={{ fontSize: '0.65rem' }}
                        >
                          CV #{comp.cvRank} → Dev #{comp.devRank}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. Final Selection Rationale Panel (Manifest Evidence) */}
      <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-xs)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <FileCode size={18} color="var(--color-accent)" />
            <h2 style={{ fontSize: 'var(--text-md)', fontWeight: 600, color: 'var(--color-ink)', margin: 0 }}>
              Frozen Model Manifest Specifications: {dataset}
            </h2>
          </div>
          <span className="chip chip-accent">
            frozen artifact: final_model_manifest.json
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--space-sm)' }}>
          <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Model Family</div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)' }}>{manifestConfig.model_family}</div>
          </div>

          <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Feature Configuration</div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)' }}>
              {manifestConfig.feature_configuration} ({manifestConfig.n_features} features)
            </div>
          </div>

          <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Target Cap / Output Constraint</div>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink)' }}>
              Cap: {manifestConfig.target_cap} ({manifestConfig.output_constraint})
            </div>
          </div>

          <div style={{ padding: 'var(--space-sm)', background: 'var(--color-paper-3)', borderRadius: 'var(--radius-sm)', border: 'var(--rule-hair) solid var(--color-rule)' }}>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>Study Identifier / Seed</div>
            <div style={{ fontSize: 'var(--text-xs)', fontFamily: 'var(--font-outlier)', color: 'var(--color-ink-2)' }}>
              {manifestConfig.tuning_study_identifier} (seed {manifestConfig.random_seed})
            </div>
          </div>
        </div>

        {/* Hyperparameters Grid */}
        <div>
          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-muted)', marginBottom: 'var(--space-2xs)' }}>
            Frozen Tuned Hyperparameters
          </div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
              gap: 'var(--space-2xs)',
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
              fontFamily: 'var(--font-outlier)',
              fontSize: 'var(--text-xs)',
            }}
          >
            {Object.entries(manifestConfig.hyperparameters).map(([key, val]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--space-xs)' }}>
                <span style={{ color: 'var(--color-muted)' }}>{key}:</span>
                <span style={{ color: 'var(--color-ink)', fontWeight: 500 }}>
                  {typeof val === 'number' ? (Number.isInteger(val) ? val : val.toFixed(4)) : String(val)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Score Progression Note */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 'var(--space-sm)',
            padding: 'var(--space-xs) var(--space-sm)',
            background: 'var(--color-paper-3)',
            borderRadius: 'var(--radius-sm)',
            border: 'var(--rule-hair) solid var(--color-rule)',
            fontSize: 'var(--text-xs)',
          }}
        >
          <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap' }}>
            <span>
              <strong style={{ color: 'var(--color-muted)' }}>Inner-CV Macro MAE:</strong>{' '}
              <span className="measured" style={{ color: 'var(--color-accent)', fontWeight: 600 }}>
                {formatCycles(manifestConfig.inner_cv_scores.engine_macro_mae, 2)} cycles
              </span>
            </span>
            <span>
              <strong style={{ color: 'var(--color-muted)' }}>Dev-Val Macro MAE:</strong>{' '}
              <span className="measured" style={{ color: 'var(--color-cool)', fontWeight: 600 }}>
                {formatCycles(manifestConfig.development_validation_scores.engine_macro_mae, 2)} cycles
              </span>
            </span>
          </div>
          <span style={{ color: 'var(--color-muted)', fontStyle: 'italic' }}>
            Official test endpoint performance is evaluated only for this frozen configuration.
          </span>
        </div>
      </div>
    </div>
  );
};
