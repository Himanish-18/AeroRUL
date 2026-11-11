import React from 'react';

interface StatTileProps {
  label: string;
  value: string | number;
  unit?: string;
  supporting?: string;
  badge?: string;
  badgeTone?: 'cool' | 'accent' | 'ok' | 'bad';
  isPrimary?: boolean;
}

export const StatTile: React.FC<StatTileProps> = ({
  label,
  value,
  unit,
  supporting,
  badge,
  badgeTone = 'cool',
  isPrimary = false,
}) => {
  return (
    <div
      className="panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minHeight: isPrimary ? '148px' : '124px',
        borderColor: isPrimary ? 'var(--color-rule-2)' : 'var(--color-rule)',
        background: isPrimary ? 'var(--color-paper-2)' : 'var(--color-paper-2)',
        padding: 'var(--space-md) var(--space-lg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-xs)' }}>
        <span
          style={{
            fontSize: 'var(--text-xs)',
            fontWeight: 500,
            color: 'var(--color-muted)',
            lineHeight: 1.2,
          }}
        >
          {label}
        </span>
        {badge && (
          <span className={`chip chip-${badgeTone}`}>
            {badge}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-2xs)', margin: 'var(--space-xs) 0' }}>
        <span
          className="measured"
          style={{
            fontSize: isPrimary ? '2.25rem' : '1.75rem',
            fontWeight: isPrimary ? 500 : 400,
            color: isPrimary ? 'var(--color-ink)' : 'var(--color-ink)',
            lineHeight: 1,
            letterSpacing: '-0.03em',
          }}
        >
          {value}
        </span>
        {unit && (
          <span
            style={{
              fontSize: 'var(--text-sm)',
              color: 'var(--color-muted)',
              fontWeight: 400,
            }}
          >
            {unit}
          </span>
        )}
      </div>

      {supporting && (
        <div
          style={{
            fontSize: 'var(--text-xs)',
            color: 'var(--color-ink-2)',
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-2xs)',
          }}
        >
          {supporting}
        </div>
      )}
    </div>
  );
};
