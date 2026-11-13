import React, { useState } from 'react';
import { Info, ChevronDown, ChevronUp } from 'lucide-react';

export const MethodologyProvenanceNotice: React.FC = () => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      style={{
        background: 'var(--color-paper-2)',
        border: 'var(--rule-hair) solid var(--color-rule-2)',
        borderRadius: 'var(--radius-card)',
        padding: 'var(--space-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-xs)',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          cursor: 'pointer',
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
          <Info size={16} color="var(--color-accent)" />
          <span style={{ fontWeight: 600, fontSize: 'var(--text-sm)', color: 'var(--color-ink)' }}>
            Evaluation Methodology & Data Provenance Guidelines
          </span>
          <span className="chip" style={{ fontSize: '0.65rem' }}>
            audited
          </span>
        </div>
        <button
          type="button"
          className="btn btn-icon btn-sm"
          style={{ border: 'none', background: 'transparent' }}
          aria-label={expanded ? 'Collapse methodology details' : 'Expand methodology details'}
        >
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)', margin: 0, lineHeight: 1.4 }}>
        This workbench strictly isolates initial validation (Phase 4), cross-validation model selection (Phase 5 Inner CV), and held-out development validation. Candidate models were never evaluated on the official NASA test set.
      </p>

      {expanded && (
        <div
          style={{
            marginTop: 'var(--space-xs)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: 'var(--space-sm)',
            fontSize: 'var(--text-xs)',
            color: 'var(--color-ink-2)',
            lineHeight: 1.45,
          }}
        >
          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--color-ink)', marginBottom: 'var(--space-3xs)' }}>
              Phase 4: Initial Benchmark
            </div>
            <div>
              Held-out 80/20 engine-level split using Phase 3.1 features with fixed starting hyperparameters and a 75-cycle window limit. Endpoint evaluation records terminal actual RUL = 0. Not official test data.
            </div>
          </div>

          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--color-ink)', marginBottom: 'var(--space-3xs)' }}>
              Phase 5: Inner Cross-Validation
            </div>
            <div>
              3-fold engine-grouped GroupKFold with positive-RUL simulated checkpoints. The declared primary model selection criterion is <code style={{ color: 'var(--color-accent)' }}>inner_cv_engine_macro_mae</code>.
            </div>
          </div>

          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--color-ink)', marginBottom: 'var(--space-3xs)' }}>
              Phase 5: Development Validation
            </div>
            <div>
              Evaluated on an independent held-out development partition. Kept strictly separate from inner-CV scores to assess rank consistency without contaminating the selection protocol.
            </div>
          </div>

          <div
            style={{
              padding: 'var(--space-sm)',
              background: 'var(--color-paper-3)',
              borderRadius: 'var(--radius-sm)',
              border: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--color-ink)', marginBottom: 'var(--space-3xs)' }}>
              Official Test & Optuna Artifact Audits
            </div>
            <div>
              Official test metrics reflect final frozen models only. The hyperparameter search artifact stores 25 trials per dataset (15 LightGBM, 10 XGBoost; 100 total), which this workbench displays without fabrication.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
