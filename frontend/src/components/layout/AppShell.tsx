import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { DatasetId } from '../../lib/data/types';

interface AppShellProps {
  selectedDataset: DatasetId;
  onSelectDataset: (dataset: DatasetId) => void;
  frozenTimestamp?: string;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
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
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed(!collapsed)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className="app-main">
        <TopBar
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

