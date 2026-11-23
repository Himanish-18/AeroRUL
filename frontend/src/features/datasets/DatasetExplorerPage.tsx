import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { DatasetId } from '../../lib/data/types';
import {
  DatasetManifest,
  DatasetSplit,
  EngineIndexRecord,
  RawObservationRecord,
} from '../../lib/data/datasetExplorerTypes';
import {
  loadDatasetManifest,
  loadEngineIndex,
  loadRawObservations,
} from '../../lib/data/datasetExplorerLoader';
import { DatasetProfileSection } from './DatasetProfileSection';
import { SplitAndEngineExplorer } from './SplitAndEngineExplorer';
import { RawObservationBrowser } from './RawObservationBrowser';
import { SensorAndSettingsExplorer } from './SensorAndSettingsExplorer';
import { SchemaAndQualitySection } from './SchemaAndQualitySection';
import {
  Database,
  Layers,
  Table,
  Activity,
  ShieldCheck,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';

interface DatasetExplorerPageProps {
  selectedDataset: DatasetId;
  onSelectDataset?: (id: DatasetId) => void;
}

type ExplorerTab = 'profile' | 'engines' | 'observations' | 'sensors' | 'quality';

export const DatasetExplorerPage: React.FC<DatasetExplorerPageProps> = ({
  selectedDataset,
  onSelectDataset,
}) => {
  const [activeTab, setActiveTab] = useState<ExplorerTab>('profile');
  const [split, setSplit] = useState<DatasetSplit>('train');
  const [selectedUnitId, setSelectedUnitId] = useState<number>(1);

  // Data states
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [engineIndex, setEngineIndex] = useState<EngineIndexRecord[]>([]);
  const [rawObservations, setRawObservations] = useState<RawObservationRecord[]>([]);

  // Loading & error states
  const [loadingManifest, setLoadingManifest] = useState<boolean>(true);
  const [, setLoadingIndex] = useState<boolean>(false);
  const [loadingRaw, setLoadingRaw] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [indexError, setIndexError] = useState<string | null>(null);

  // Load manifest once
  useEffect(() => {
    let isMounted = true;
    setLoadingManifest(true);
    setError(null);

    loadDatasetManifest()
      .then((data) => {
        if (isMounted) {
          setManifest(data);
          setLoadingManifest(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error('[DatasetExplorerPage] Failed to load manifest:', err);
          setError(err.message || 'Failed to load dataset explorer manifest');
          setLoadingManifest(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Fetch engine index for the current dataset and split
  const fetchEngineIndex = useCallback(() => {
    let isMounted = true;
    setLoadingIndex(true);
    setIndexError(null);
    // Crucial: Clear stale records so previous dataset/split records are not retained
    setEngineIndex([]);

    loadEngineIndex(selectedDataset, split)
      .then((records) => {
        if (isMounted) {
          setEngineIndex(records);
          setLoadingIndex(false);
          setIndexError(null);
          // If selectedUnitId is out of bounds for the new index, reset to first unit
          if (!records.some((r) => r.unit_id === selectedUnitId)) {
            setSelectedUnitId(records[0]?.unit_id || 1);
          }
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error('[DatasetExplorerPage] Failed to load engine index:', err);
          setEngineIndex([]);
          setIndexError(err.message || `Failed to load engine index for ${selectedDataset} (${split})`);
          setLoadingIndex(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDataset, split, selectedUnitId]);

  // Load engine index when dataset or split changes
  useEffect(() => {
    const cleanup = fetchEngineIndex();
    return cleanup;
  }, [fetchEngineIndex]);

  // Lazily load raw observations when observations or sensors tab is active
  useEffect(() => {
    if (activeTab !== 'observations' && activeTab !== 'sensors') {
      return;
    }

    let isMounted = true;
    setLoadingRaw(true);

    loadRawObservations(selectedDataset, split)
      .then((obs) => {
        if (isMounted) {
          setRawObservations(obs);
          setLoadingRaw(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          console.error('[DatasetExplorerPage] Failed to load raw observations:', err);
          setError(err.message || 'Failed to decompress and load raw observations');
          setLoadingRaw(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [selectedDataset, split, activeTab]);

  const availableUnits = useMemo(() => {
    return engineIndex.map((e) => e.unit_id);
  }, [engineIndex]);

  const currentDatasetProfile = manifest?.datasets[selectedDataset];
  const splitSummary = currentDatasetProfile ? currentDatasetProfile[split] : null;

  const handleNavigateToSplit = useCallback((newSplit: DatasetSplit) => {
    setSplit(newSplit);
    setActiveTab('engines');
  }, []);

  const handleInspectEngineInRaw = useCallback((unitId: number) => {
    setSelectedUnitId(unitId);
    setActiveTab('observations');
  }, []);

  if (loadingManifest) {
    return (
      <div className="workspace-loading" style={{ padding: '64px', textAlign: 'center' }}>
        <div style={{ color: 'var(--color-accent)', marginBottom: 'var(--space-sm)' }}>
          <Database size={32} />
        </div>
        <div style={{ fontSize: 'var(--text-md)', color: 'var(--color-ink)' }}>
          Loading NASA C-MAPSS Dataset Explorer…
        </div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
          Reading canonical schema, metadata, and engine indexes for {selectedDataset}
        </div>
      </div>
    );
  }

  if (error || !manifest || !currentDatasetProfile) {
    return (
      <div className="panel" style={{ margin: 'var(--space-lg)', borderLeft: '3px solid var(--color-bad)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', color: 'var(--color-bad)' }}>
          <AlertCircle size={20} />
          <h3 style={{ margin: 0, fontSize: 'var(--text-md)' }}>Error Loading Dataset Explorer</h3>
        </div>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink-2)', margin: 'var(--space-xs) 0' }}>
          {error || `Dataset profile not found for ${selectedDataset}`}
        </p>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={() => window.location.reload()}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
        >
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)', paddingBottom: 'var(--space-2xl)' }}>
      {/* Header and Section Switcher */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 'var(--space-md)',
          borderBottom: 'var(--rule-hair) solid var(--color-rule)',
          paddingBottom: 'var(--space-md)',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-xs)' }}>
            <Database size={20} style={{ color: 'var(--color-accent)' }} />
            <h2 style={{ fontSize: 'var(--text-xl)', color: 'var(--color-ink)', margin: 0 }}>
              Dataset Explorer: {selectedDataset}
            </h2>
            <span className="chip chip-ok" style={{ fontSize: 'var(--text-xs)', marginLeft: '4px' }}>
              Canonical 26-Col Schema
            </span>
          </div>
          <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-muted)' }}>
            NASA C-MAPSS raw observations, engine indexes, and dataset profiling
          </span>
        </div>

        {/* 5-Section Tab Navigation */}
        <div className="segmented-control" role="tablist">
          <button
            type="button"
            className={`segmented-button ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
            role="tab"
            aria-selected={activeTab === 'profile'}
            id="tab-dataset-profile"
          >
            <Database size={14} style={{ marginRight: '4px' }} />
            Profile
          </button>
          <button
            type="button"
            className={`segmented-button ${activeTab === 'engines' ? 'active' : ''}`}
            onClick={() => setActiveTab('engines')}
            role="tab"
            aria-selected={activeTab === 'engines'}
            id="tab-engine-index"
          >
            <Layers size={14} style={{ marginRight: '4px' }} />
            Engines
          </button>
          <button
            type="button"
            className={`segmented-button ${activeTab === 'observations' ? 'active' : ''}`}
            onClick={() => setActiveTab('observations')}
            role="tab"
            aria-selected={activeTab === 'observations'}
            id="tab-raw-observations"
          >
            <Table size={14} style={{ marginRight: '4px' }} />
            Raw Observations
          </button>
          <button
            type="button"
            className={`segmented-button ${activeTab === 'sensors' ? 'active' : ''}`}
            onClick={() => setActiveTab('sensors')}
            role="tab"
            aria-selected={activeTab === 'sensors'}
            id="tab-sensor-explorer"
          >
            <Activity size={14} style={{ marginRight: '4px' }} />
            Sensors & Settings
          </button>
          <button
            type="button"
            className={`segmented-button ${activeTab === 'quality' ? 'active' : ''}`}
            onClick={() => setActiveTab('quality')}
            role="tab"
            aria-selected={activeTab === 'quality'}
            id="tab-data-quality"
          >
            <ShieldCheck size={14} style={{ marginRight: '4px' }} />
            Quality & Schema
          </button>
        </div>
      </div>

      {/* SECTION 1: DATASET PROFILE */}
      {activeTab === 'profile' && (
        <DatasetProfileSection
          manifest={manifest}
          selectedDataset={selectedDataset}
          onSelectDataset={onSelectDataset || (() => {})}
          onNavigateToSplit={handleNavigateToSplit}
        />
      )}

      {/* SECTION 2: SPLIT & ENGINE EXPLORER */}
      {activeTab === 'engines' && (
        indexError ? (
          <div
            className="panel"
            role="alert"
            aria-live="polite"
            style={{
              borderLeft: '3px solid var(--color-bad)',
              padding: 'var(--space-md)',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-sm)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', color: 'var(--color-bad)' }}>
              <AlertCircle size={20} />
              <h4 style={{ margin: 0, fontSize: 'var(--text-md)' }}>Engine Index Load Error</h4>
            </div>
            <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-ink-2)', margin: 0 }}>
              {indexError}
            </p>
            <div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={fetchEngineIndex}
                style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', whiteSpace: 'nowrap' }}
                id="btn-retry-engine-index"
              >
                <RefreshCw size={14} /> Retry engine index
              </button>
            </div>
          </div>
        ) : (
          <SplitAndEngineExplorer
            dataset={selectedDataset}
            split={split}
            onSelectSplit={setSplit}
            engineIndex={engineIndex}
            selectedUnitId={selectedUnitId}
            onSelectEngine={setSelectedUnitId}
            onInspectEngineInRaw={handleInspectEngineInRaw}
          />
        )
      )}

      {/* SECTION 3: RAW OBSERVATION BROWSER */}
      {activeTab === 'observations' && (
        <RawObservationBrowser
          dataset={selectedDataset}
          split={split}
          onSelectSplit={setSplit}
          availableUnits={availableUnits}
          selectedUnitId={selectedUnitId}
          onSelectUnit={setSelectedUnitId}
          rawObservations={rawObservations}
          isLoading={loadingRaw}
          sensorSpecs={manifest.sensor_specifications}
        />
      )}

      {/* SECTION 4: SENSORS & SETTINGS EXPLORER */}
      {activeTab === 'sensors' && splitSummary && (
        <SensorAndSettingsExplorer
          dataset={selectedDataset}
          split={split}
          onSelectSplit={setSplit}
          rawObservations={rawObservations}
          sensorSummaries={splitSummary.sensors}
          opSettingSummaries={splitSummary.operating_settings}
          sensorSpecs={manifest.sensor_specifications}
          rulStats={splitSummary.rul_statistics}
          engineIndex={engineIndex}
        />
      )}

      {/* SECTION 5: SCHEMA & DATA QUALITY */}
      {activeTab === 'quality' && splitSummary && (
        <SchemaAndQualitySection
          dataset={selectedDataset}
          split={split}
          qualityStats={splitSummary.quality}
          sensorSpecs={manifest.sensor_specifications}
          opSettingSpecs={manifest.operating_setting_specifications}
        />
      )}
    </div>
  );
};
