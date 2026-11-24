import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { DatasetId, AppView } from '../../lib/data/types';

interface AppShellProps {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
  selectedDataset: DatasetId;
  onSelectDataset: (dataset: DatasetId) => void;
  frozenTimestamp?: string;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentView,
  onNavigate,
  selectedDataset,
  onSelectDataset,
  frozenTimestamp,
  children,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-container">
      <Sidebar
        currentView={currentView}
        onNavigate={onNavigate}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className="app-main">
        <TopBar
          currentView={currentView}
          selectedDataset={selectedDataset}
          onSelectDataset={onSelectDataset}
          frozenTimestamp={frozenTimestamp}
          onOpenMobileMenu={() => setMobileOpen(true)}
        />

        <main className="app-content">
          {children}
        </main>
      </div>
    </div>
  );
};

