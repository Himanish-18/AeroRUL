import React from 'react';

interface SkeletonProps {
  className?: string;
  height?: string | number;
  width?: string | number;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '', height = '20px', width = '100%' }) => {
  return (
    <div
      className={`skeleton-pulse ${className}`}
      style={{
        height,
        width,
        background: 'var(--color-rule)',
        borderRadius: 'var(--radius-pill)',
        opacity: 0.6,
      }}
      aria-hidden="true"
    />
  );
};

export const DashboardSkeleton: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
      {/* Top metrics skeleton row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--space-md)' }}>
        <div className="panel" style={{ minHeight: '130px', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <Skeleton width="40%" height="16px" />
          <Skeleton width="70%" height="36px" />
          <Skeleton width="50%" height="14px" />
        </div>
        <div className="panel" style={{ minHeight: '130px', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <Skeleton width="40%" height="16px" />
          <Skeleton width="60%" height="36px" />
          <Skeleton width="50%" height="14px" />
        </div>
        <div className="panel" style={{ minHeight: '130px', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <Skeleton width="40%" height="16px" />
          <Skeleton width="60%" height="36px" />
          <Skeleton width="50%" height="14px" />
        </div>
        <div className="panel" style={{ minHeight: '130px', display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)' }}>
          <Skeleton width="40%" height="16px" />
          <Skeleton width="60%" height="36px" />
          <Skeleton width="50%" height="14px" />
        </div>
      </div>

      {/* Main visualization skeleton */}
      <div className="panel" style={{ height: '420px', display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <Skeleton width="30%" height="24px" />
        <Skeleton width="100%" height="320px" />
      </div>

      {/* Table skeleton */}
      <div className="panel" style={{ height: '300px', display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
        <Skeleton width="25%" height="24px" />
        <Skeleton width="100%" height="200px" />
      </div>
    </div>
  );
};

