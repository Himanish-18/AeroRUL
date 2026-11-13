import React from 'react';
import { Binary, ShieldCheck, Layers } from 'lucide-react';
import { DatasetId, ModelEvaluationTab } from '../../lib/data/types';

interface ModelEvaluationHeaderProps {
  selectedDataset: DatasetId;
  activeTab: ModelEvaluationTab;
  onTabChange: (tab: ModelEvaluationTab) => void;
  candidateCount: number;
  frozenModel: string;
  selectionCriterion: string;
}

export const ModelEvaluationHeader: React.FC<ModelEvaluationHeaderProps> = ({
  selectedDataset,
  activeTab,
  onTabChange,
  candidateCount,
  frozenModel,
  selectionCriterion,
}) => {
  const tabs: { id: ModelEvaluationTab; label: string; icon: React.ReactNode }[] = [
    {
      id: 'phase5_selection',
      label: 'Phase 5: Model Selection',
      icon: <ShieldCheck size={14} />,
    },
    {
      id: 'phase4_benchmark',
      label: 'Phase 4: Initial Benchmark',
      icon: <Binary size={14} />,
    },
    {
      id: 'configuration_tuning',
      label: 'Configuration & Tuning',
      icon: <Layers size={14} />,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
      {/* Title & Metadata row */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 'var(--space-md)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3xs)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', flexWrap: 'wrap' }}>
            <h1
              style={{
                fontSize: 'var(--text-xl)',
                fontFamily: 'var(--font-display)',
                fontWeight: 600,
                color: 'var(--color-ink)',
                letterSpacing: 'var(--tracking-display)',
                margin: 0,
              }}
            >
              Model Evaluation
            </h1>
            <span className="chip chip-accent" style={{ fontWeight: 600 }}>
              {selectedDataset}
            </span>
            <span className="chip" style={{ color: 'var(--color-ink-2)' }}>
              {candidateCount} candidates
            </span>
            <span className="chip" style={{ color: 'var(--color-ink-2)' }}>
              frozen: {frozenModel}
            </span>
          </div>
          <p
            style={{
              fontSize: 'var(--text-sm)',
              color: 'var(--color-muted)',
              margin: 0,
              maxWidth: '65ch',
              lineHeight: 1.4,
            }}
          >
            Candidate model benchmarking, cross-validation evidence, development validation, and frozen model selection.
          </p>
        </div>

        {/* Selection Criterion Pill */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-2xs)',
            padding: 'var(--space-2xs) var(--space-sm)',
            background: 'var(--color-paper-2)',
            border: 'var(--rule-hair) solid var(--color-rule-2)',
            borderRadius: 'var(--radius-card)',
            fontSize: 'var(--text-xs)',
          }}
        >
          <span style={{ color: 'var(--color-muted)' }}>selection criterion:</span>
          <code
            style={{
              color: 'var(--color-accent)',
              fontFamily: 'var(--font-outlier)',
              fontWeight: 500,
            }}
          >
            {selectionCriterion}
          </code>
        </div>
      </div>

      {/* Segmented Tab Bar */}
      <div
        role="tablist"
        aria-label="Evaluation comparison contexts"
        style={{
          display: 'inline-flex',
          padding: 'var(--space-3xs)',
          background: 'var(--color-paper-3)',
          border: 'var(--rule-hair) solid var(--color-rule)',
          borderRadius: 'var(--radius-pill)',
          gap: 'var(--space-3xs)',
          maxWidth: '100%',
          overflowX: 'auto',
        }}
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              role="tab"
              type="button"
              aria-selected={isActive}
              onClick={() => onTabChange(tab.id)}
              className="btn btn-sm"
              style={{
                background: isActive ? 'var(--color-paper-2)' : 'transparent',
                borderColor: isActive ? 'var(--color-rule-2)' : 'transparent',
                color: isActive ? 'var(--color-ink)' : 'var(--color-ink-2)',
                fontWeight: isActive ? 600 : 400,
                borderRadius: 'var(--radius-pill)',
                whiteSpace: 'nowrap',
                padding: '0 var(--space-md)',
                height: '32px',
                cursor: 'pointer',
              }}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
