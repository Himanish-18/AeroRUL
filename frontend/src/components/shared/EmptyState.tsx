import React from 'react';
import { SearchX } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  actionText?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No engine records found',
  description = 'No engines match the current search query or filter criteria.',
  actionText,
  onAction,
}) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 'var(--space-2xl) var(--space-md)',
        textAlign: 'center',
        color: 'var(--color-muted)',
      }}
    >
      <SearchX size={32} color="var(--color-rule-2)" style={{ marginBottom: 'var(--space-md)' }} />
      <h3 style={{ fontSize: 'var(--text-title)', color: 'var(--color-ink-2)', marginBottom: 'var(--space-2xs)' }}>
        {title}
      </h3>
      <p style={{ fontSize: 'var(--text-sm)', maxWidth: '40ch', marginBottom: 'var(--space-md)' }}>
        {description}
      </p>
      {actionText && onAction && (
        <button type="button" className="btn btn-sm" onClick={onAction}>
          {actionText}
        </button>
      )}
    </div>
  );
};

