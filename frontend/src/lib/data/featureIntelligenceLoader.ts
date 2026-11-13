/**
 * AeroRUL — Feature Intelligence & Degradation Data Loader
 *
 * Dedicated loader for Stage 4 interpretability and trajectory artifacts.
 * Kept strictly isolated so any failure here never cascades to Stage 1, 2, or 3.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  DatasetId,
  FeatureImportanceRecord,
  SensorImportanceRecord,
  FeatureFamilyRecord,
  SensorTrajectoryRecord,
  FeatureImportanceManifest,
  SensorTrajectoriesManifest,
} from './types';
import {
  parseFeatureImportanceCsv,
  parseSensorImportanceCsv,
  parseFeatureFamilyCsv,
  parseSensorTrajectoriesCsv,
  parseFeatureImportanceManifest,
  parseSensorTrajectoriesManifest,
} from './parser';

export interface LoadedFeatureIntelligenceData {
  featureImportance: Record<DatasetId, FeatureImportanceRecord[]>;
  sensorImportance: Record<DatasetId, SensorImportanceRecord[]>;
  familyBreakdown: FeatureFamilyRecord[];
  featureManifest: FeatureImportanceManifest;
  trajectoriesManifest: SensorTrajectoriesManifest;
  syncedTimestamp: string;
}

let cachedCoreData: LoadedFeatureIntelligenceData | null = null;
let inflightCorePromise: Promise<LoadedFeatureIntelligenceData> | null = null;
const cachedTrajectories: Partial<Record<DatasetId, SensorTrajectoryRecord[]>> = {};

function getArtifactUrl(filename: string): string {
  const base = import.meta.env.BASE_URL || './';
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  return `${cleanBase}data/stage4/${filename}`;
}

async function fetchText(url: string, label: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${label} from ${url} (HTTP ${res.status}: ${res.statusText})`);
  }
  return res.text();
}

async function fetchJson<T>(url: string, label: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${label} from ${url} (HTTP ${res.status}: ${res.statusText})`);
  }
  return res.json() as Promise<T>;
}

export async function loadFeatureIntelligenceCoreData(
  forceReload = false
): Promise<LoadedFeatureIntelligenceData> {
  if (cachedCoreData && !forceReload) {
    return cachedCoreData;
  }
  if (inflightCorePromise && !forceReload) {
    return inflightCorePromise;
  }

  inflightCorePromise = (async () => {
    try {
      const [
        featFd001Text,
        featFd002Text,
        featFd003Text,
        featFd004Text,
        sensorFd001Text,
        sensorFd002Text,
        sensorFd003Text,
        sensorFd004Text,
        familyText,
        featManifestJson,
        trajManifestJson,
      ] = await Promise.all([
        fetchText(getArtifactUrl('feature_importance_FD001.csv'), 'FD001 feature importance'),
        fetchText(getArtifactUrl('feature_importance_FD002.csv'), 'FD002 feature importance'),
        fetchText(getArtifactUrl('feature_importance_FD003.csv'), 'FD003 feature importance'),
        fetchText(getArtifactUrl('feature_importance_FD004.csv'), 'FD004 feature importance'),
        fetchText(getArtifactUrl('sensor_importance_FD001.csv'), 'FD001 sensor importance'),
        fetchText(getArtifactUrl('sensor_importance_FD002.csv'), 'FD002 sensor importance'),
        fetchText(getArtifactUrl('sensor_importance_FD003.csv'), 'FD003 sensor importance'),
        fetchText(getArtifactUrl('sensor_importance_FD004.csv'), 'FD004 sensor importance'),
        fetchText(getArtifactUrl('feature_family_importance.csv'), 'feature family breakdown'),
        fetchJson(getArtifactUrl('feature_importance_manifest.json'), 'feature importance manifest'),
        fetchJson(getArtifactUrl('sensor_trajectories_manifest.json'), 'sensor trajectories manifest'),
      ]);

      const data: LoadedFeatureIntelligenceData = {
        featureImportance: {
          FD001: parseFeatureImportanceCsv(featFd001Text, 'FD001'),
          FD002: parseFeatureImportanceCsv(featFd002Text, 'FD002'),
          FD003: parseFeatureImportanceCsv(featFd003Text, 'FD003'),
          FD004: parseFeatureImportanceCsv(featFd004Text, 'FD004'),
        },
        sensorImportance: {
          FD001: parseSensorImportanceCsv(sensorFd001Text, 'FD001'),
          FD002: parseSensorImportanceCsv(sensorFd002Text, 'FD002'),
          FD003: parseSensorImportanceCsv(sensorFd003Text, 'FD003'),
          FD004: parseSensorImportanceCsv(sensorFd004Text, 'FD004'),
        },
        familyBreakdown: parseFeatureFamilyCsv(familyText),
        featureManifest: parseFeatureImportanceManifest(featManifestJson),
        trajectoriesManifest: parseSensorTrajectoriesManifest(trajManifestJson),
        syncedTimestamp: new Date().toISOString(),
      };

      cachedCoreData = data;
      return data;
    } finally {
      inflightCorePromise = null;
    }
  })();

  return inflightCorePromise;
}

export async function loadSensorTrajectoriesData(
  dataset: DatasetId,
  forceReload = false
): Promise<SensorTrajectoryRecord[]> {
  if (cachedTrajectories[dataset] && !forceReload) {
    return cachedTrajectories[dataset]!;
  }

  const csvText = await fetchText(
    getArtifactUrl(`sensor_trajectories_${dataset}.csv`),
    `${dataset} sensor trajectories`
  );
  const parsed = parseSensorTrajectoriesCsv(csvText, dataset);
  cachedTrajectories[dataset] = parsed;
  return parsed;
}

export interface UseFeatureIntelligenceResult {
  data: LoadedFeatureIntelligenceData | null;
  trajectories: SensorTrajectoryRecord[] | null;
  isLoading: boolean;
  isLoadingTrajectories: boolean;
  error: string | null;
  reload: () => void;
}

export function useFeatureIntelligenceData(selectedDataset: DatasetId): UseFeatureIntelligenceResult {
  const [data, setData] = useState<LoadedFeatureIntelligenceData | null>(cachedCoreData);
  const [trajectories, setTrajectories] = useState<SensorTrajectoryRecord[] | null>(
    cachedTrajectories[selectedDataset] || null
  );
  const [isLoading, setIsLoading] = useState<boolean>(!cachedCoreData);
  const [isLoadingTrajectories, setIsLoadingTrajectories] = useState<boolean>(
    !cachedTrajectories[selectedDataset]
  );
  const [error, setError] = useState<string | null>(null);

  const fetchAll = useCallback(
    async (forceReload = false) => {
      setIsLoading(true);
      setError(null);
      try {
        const core = await loadFeatureIntelligenceCoreData(forceReload);
        setData(core);

        setIsLoadingTrajectories(true);
        const traj = await loadSensorTrajectoriesData(selectedDataset, forceReload);
        setTrajectories(traj);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
        setIsLoadingTrajectories(false);
      }
    },
    [selectedDataset]
  );

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        if (!data) {
          setIsLoading(true);
          const core = await loadFeatureIntelligenceCoreData(false);
          if (!cancelled) setData(core);
        }

        setIsLoadingTrajectories(true);
        const traj = await loadSensorTrajectoriesData(selectedDataset, false);
        if (!cancelled) setTrajectories(traj);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
          setIsLoadingTrajectories(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [selectedDataset, data]);

  return {
    data,
    trajectories,
    isLoading,
    isLoadingTrajectories,
    error,
    reload: () => fetchAll(true),
  };
}
