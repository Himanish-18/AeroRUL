import React from 'react';
import { DatasetId } from '../../lib/data/types';
import { AlertTriangle, Info, ShieldCheck } from 'lucide-react';

interface DegradationInterpretationPanelProps {
  dataset: DatasetId;
}

export const DegradationInterpretationPanel: React.FC<DegradationInterpretationPanelProps> = ({
  dataset,
}) => {
  const isMultiRegime = dataset === 'FD002' || dataset === 'FD004';

  return (
    <div
      className="card"
      style={{
        padding: 'var(--space-md) var(--space-lg)',
        background: 'var(--color-paper-2)',
        border: 'var(--rule-hair) solid var(--color-rule)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-sm)',
      }}
      data-testid="degradation-interpretation-panel"
      aria-label="Degradation interpretation and domain guidance"
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
        <ShieldCheck size={16} style={{ color: 'var(--color-accent)' }} />
        <h4 style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink)', margin: 0, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Telemetry Interpretation & Methodological Boundaries
        </h4>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-md)', fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
        {/* Boundary 1: No Synthetic Health Index */}
        <div style={{ display: 'flex', gap: 'var(--space-xs)', alignItems: 'flex-start' }}>
          <Info size={14} style={{ color: 'var(--color-cool)', flexShrink: 0, marginTop: 2 }} />
          <div>
            <strong style={{ color: 'var(--color-ink)' }}>Authentic Telemetry (No Fabricated Health Index):</strong>
            <p style={{ margin: '2px 0 0 0', lineHeight: 1.4 }}>
              The curves above represent authentic, un-interpolated NASA C-MAPSS training measurements. No artificial health index, smoothing spline, or synthetic proxy was synthesized. Variations represent genuine physical sensor readings under aerodynamic degradation.
            </p>
          </div>
        </div>

        {/* Boundary 2: Multi-Regime Operational Variation */}
        {isMultiRegime ? (
          <div style={{ display: 'flex', gap: 'var(--space-xs)', alignItems: 'flex-start' }}>
            <AlertTriangle size={14} style={{ color: 'var(--color-accent)', flexShrink: 0, marginTop: 2 }} />
            <div>
              <strong style={{ color: 'var(--color-accent)' }}>Multi-Regime Operating Condition Note ({dataset}):</strong>
              <p style={{ margin: '2px 0 0 0', lineHeight: 1.4 }}>
                {dataset} operates under <strong>six distinct operating conditions</strong> (altitude 0 to 42,000 ft, Mach 0.0 to 0.84). Raw sensor shifts reflect aerodynamic regime changes superimposed on progressive physical wear. LightGBM models handle this through regime-specific normalization and sliding window statistics.
              </p>
            </div>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 'var(--space-xs)', alignItems: 'flex-start' }}>
            <Info size={14} style={{ color: 'var(--color-ok)', flexShrink: 0, marginTop: 2 }} />
            <div>
              <strong style={{ color: 'var(--color-ink)' }}>Single-Regime Sea Level Condition ({dataset}):</strong>
              <p style={{ margin: '2px 0 0 0', lineHeight: 1.4 }}>
                {dataset} operates at sea level under single stationary operating conditions. Sensor drift across cycles directly reflects progressive mechanical degradation (HPC clearance widening and thermal deterioration) contaminated by Gaussian measurement noise.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
