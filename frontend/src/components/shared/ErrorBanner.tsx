import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = 'Evaluation data could not be loaded',
  message,
  onRetry,
}) => {
  return (
    <div
      className="panel"
      style={{
        borderColor: 'var(--color-bad)',
        backgroundColor: 'var(--color-bad-soft)',
        display: 'flex',
        flexDirection: 'column',
        gap: 'var(--space-sm)',
        padding: 'var(--space-lg)',
      }}
      role="alert"
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
        <AlertCircle size={20} color="var(--color-bad)" style={{ flexShrink: 0 }} />
        <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink)', margin: 0 }}>
          {title}
        </h3>
      </div>
      <p style={{ color: 'var(--color-ink-2)', fontSize: 'var(--text-sm)', margin: 0 }}>
        {message}
      </p>
      {onRetry && (
        <div style={{ marginTop: 'var(--space-xs)' }}>
          <button
            type="button"
            className="btn btn-sm"
            onClick={onRetry}
            style={{
              borderColor: 'var(--color-bad)',
              color: 'var(--color-ink)',
            }}
          >
            <RefreshCw size={14} />
            <span>Retry loading artifacts</span>
          </button>
        </div>
      )}
    </div>
  );
};

