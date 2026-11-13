import React from 'react';
import {
  LayoutDashboard,
  Gauge,
  Activity,
  Binary,
  GitCompare,
  Database,
  FileText,
  ChevronLeft,
  ChevronRight,
  Plane,
} from 'lucide-react';
import { AppView } from '../../lib/data/types';

interface SidebarProps {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  status: 'active' | 'planned';
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: <LayoutDashboard size={18} />,
    status: 'active',
  },
  {
    id: 'predictions',
    label: 'Engine Predictions',
    icon: <Gauge size={18} />,
    status: 'active',
  },
  {
    id: 'sensors',
    label: 'Sensor Analytics',
    icon: <Activity size={18} />,
    status: 'planned',
  },
  {
    id: 'models',
    label: 'Model Evaluation',
    icon: <Binary size={18} />,
    status: 'active',
  },
  {
    id: 'experiments',
    label: 'Experiment Comparison',
    icon: <GitCompare size={18} />,
    status: 'planned',
  },
  {
    id: 'datasets',
    label: 'Dataset Explorer',
    icon: <Database size={18} />,
    status: 'planned',
  },
  {
    id: 'reports',
    label: 'Research Reports',
    icon: <FileText size={18} />,
    status: 'planned',
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onNavigate,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
}) => {
  return (
    <>
      {mobileOpen && (
        <div
          onClick={onCloseMobile}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'var(--color-backdrop)',
            zIndex: 'calc(var(--z-modal) - 1)',
          }}
          aria-hidden="true"
        />
      )}
      <aside className={`app-sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}>
        {/* Brand Header */}
        <div
          style={{
            height: '56px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'space-between',
            padding: collapsed ? '0' : '0 var(--space-md)',
            borderBottom: 'var(--rule-hair) solid var(--color-rule)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <div
              style={{
                width: '28px',
                height: '28px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--color-accent-soft)',
                border: 'var(--rule-hair) solid var(--color-accent)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--color-accent)',
                flexShrink: 0,
              }}
            >
              <Plane size={16} />
            </div>
            {!collapsed && (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span
                  style={{
                    fontFamily: 'var(--font-display)',
                    fontWeight: 500,
                    fontSize: 'var(--text-md)',
                    color: 'var(--color-ink)',
                    lineHeight: 1.1,
                    letterSpacing: 'var(--tracking-display)',
                  }}
                >
                  AeroRUL
                </span>
                <span
                  style={{
                    fontSize: 'var(--text-xs)',
                    color: 'var(--color-muted)',
                    lineHeight: 1,
                  }}
                >
                  turbofan RUL lab
                </span>
              </div>
            )}
          </div>

          {!collapsed && (
            <button
              type="button"
              className="btn btn-icon btn-sm"
              onClick={onToggleCollapse}
              aria-label="Collapse sidebar"
              style={{ border: 'none', background: 'transparent' }}
            >
              <ChevronLeft size={16} color="var(--color-muted)" />
            </button>
          )}
        </div>

        {/* Navigation destinations */}
        <nav
          style={{
            flex: 1,
            padding: 'var(--space-md) var(--space-xs)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-2xs)',
          }}
          aria-label="Main navigation"
        >
          {NAV_ITEMS.map((item) => {
            const isPlanned = item.status === 'planned';
            const isActive = currentView === item.id;

            const handleClick = () => {
              if (!isPlanned && item.id !== currentView) {
                onNavigate(item.id as AppView);
                onCloseMobile();
              }
            };

            return (
              <div
                key={item.id}
                role="link"
                tabIndex={isPlanned ? -1 : 0}
                aria-disabled={isPlanned ? 'true' : undefined}
                aria-current={isActive ? 'page' : undefined}
                title={collapsed ? `${item.label}${isPlanned ? ' (Planned)' : ''}` : undefined}
                onClick={handleClick}
                onKeyDown={(e) => {
                  if (!isPlanned && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    handleClick();
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-xs)',
                  height: '38px',
                  padding: collapsed ? '0' : '0 var(--space-sm)',
                  justifyContent: collapsed ? 'center' : 'space-between',
                  borderRadius: 'var(--radius-pill)',
                  background: isActive ? 'var(--color-paper-2)' : 'transparent',
                  border: isActive ? 'var(--rule-hair) solid var(--color-rule-2)' : 'var(--rule-hair) solid transparent',
                  color: isActive ? 'var(--color-ink)' : isPlanned ? 'var(--color-muted)' : 'var(--color-ink-2)',
                  cursor: isPlanned ? 'not-allowed' : 'pointer',
                  opacity: isPlanned ? 0.65 : 1,
                  position: 'relative',
                  outline: 'none',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)', minWidth: 0 }}>
                  <span
                    style={{
                      color: isActive ? 'var(--color-accent)' : 'inherit',
                      display: 'flex',
                      alignItems: 'center',
                      flexShrink: 0,
                    }}
                  >
                    {item.icon}
                  </span>
                  {!collapsed && (
                    <span
                      style={{
                        fontSize: 'var(--text-sm)',
                        fontWeight: isActive ? 500 : 400,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {item.label}
                    </span>
                  )}
                </div>

                {!collapsed && isPlanned && (
                  <span
                    style={{
                      fontSize: '0.65rem',
                      textTransform: 'lowercase',
                      padding: '1px 5px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--color-paper)',
                      border: 'var(--rule-hair) solid var(--color-rule)',
                      color: 'var(--color-muted)',
                      flexShrink: 0,
                    }}
                  >
                    planned
                  </span>
                )}
              </div>
            );
          })}
        </nav>

        {/* Footer / Toggle for collapsed state */}
        {collapsed && (
          <div
            style={{
              padding: 'var(--space-xs)',
              display: 'flex',
              justifyContent: 'center',
              borderTop: 'var(--rule-hair) solid var(--color-rule)',
            }}
          >
            <button
              type="button"
              className="btn btn-icon btn-sm"
              onClick={onToggleCollapse}
              aria-label="Expand sidebar"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        )}

        {!collapsed && (
          <div
            style={{
              padding: 'var(--space-sm) var(--space-md)',
              borderTop: 'var(--rule-hair) solid var(--color-rule)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-muted)',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2xs)' }}>
              <span style={{ color: 'var(--color-cool)' }}>Benchmark</span>
              <span>Stage 2</span>
            </div>
            <span style={{ fontSize: '0.7rem' }}>NASA C-MAPSS FD001-FD004</span>
          </div>
        )}
      </aside>
    </>
  );
};

