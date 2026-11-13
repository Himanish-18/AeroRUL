/**
 * AeroRUL — Model Evaluation Data Loader
 *
 * Dedicated loader for candidate model comparison, cross-validation,
 * development validation, sensitivity studies, and hyperparameter trials.
 * Kept strictly isolated from the Stage 1 & 2 loaders so evaluation workbench
 * failures never cascade to the Overview or Engine Predictions screens.
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Phase4BenchmarkRecord,
  Phase5CrossValidationRecord,
  Phase5DevelopmentValidationRecord,
  Phase5WindowAblationRecord,
  Phase5TargetCapSensitivityRecord,
  HyperparameterTrialRecord,
  ModelManifest,
} from './types';
import {
  parsePhase4BenchmarkCsv,
  parsePhase5CrossValidationCsv,
  parsePhase5DevelopmentValidationCsv,
  parsePhase5WindowAblationCsv,
  parsePhase5TargetCapSensitivityCsv,
  parseHyperparameterSearchCsv,
  parseModelManifest,
} from './parser';

export interface LoadedModelEvaluationData {
  phase4Benchmark: Phase4BenchmarkRecord[];
  phase5CrossValidation: Phase5CrossValidationRecord[];
  phase5DevelopmentValidation: Phase5DevelopmentValidationRecord[];
  phase5WindowAblation: Phase5WindowAblationRecord[];
  phase5TargetCapSensitivity: Phase5TargetCapSensitivityRecord[];
  hyperparameterTrials: HyperparameterTrialRecord[];
  manifest: ModelManifest;
  syncedTimestamp: string;
}

let cachedData: LoadedModelEvaluationData | null = null;
let inflightPromise: Promise<LoadedModelEvaluationData> | null = null;

function getArtifactUrl(phase: 'phase4' | 'phase5', filename: string): string {
  const base = import.meta.env.BASE_URL || './';
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  return `${cleanBase}data/${phase}/${filename}`;
}

export async function loadModelEvaluationData(forceReload = false): Promise<LoadedModelEvaluationData> {
  if (cachedData && !forceReload) {
    return cachedData;
  }

  if (inflightPromise && !forceReload) {
    return inflightPromise;
  }

  inflightPromise = (async () => {
    try {
      const p4BenchmarkUrl = getArtifactUrl('phase4', 'model_benchmark_metrics.csv');
      const p5CvUrl = getArtifactUrl('phase5', 'cross_validation_metrics.csv');
      const p5DevUrl = getArtifactUrl('phase5', 'development_validation_metrics.csv');
      const p5WindowUrl = getArtifactUrl('phase5', 'window_ablation_metrics.csv');
      const p5CapUrl = getArtifactUrl('phase5', 'target_cap_sensitivity.csv');
      const p5HyperUrl = getArtifactUrl('phase5', 'hyperparameter_search_results.csv');
      const manifestUrl = getArtifactUrl('phase5', 'final_model_manifest.json');

      const [
        p4BenchmarkRes,
        p5CvRes,
        p5DevRes,
        p5WindowRes,
        p5CapRes,
        p5HyperRes,
        manifestRes,
      ] = await Promise.all([
        fetch(p4BenchmarkUrl),
        fetch(p5CvUrl),
        fetch(p5DevUrl),
        fetch(p5WindowUrl),
        fetch(p5CapUrl),
        fetch(p5HyperUrl),
        fetch(manifestUrl),
      ]);

      if (!p4BenchmarkRes.ok) {
        throw new Error(`Failed to load Phase 4 benchmark artifact (${p4BenchmarkRes.status} ${p4BenchmarkRes.statusText})`);
      }
      if (!p5CvRes.ok) {
        throw new Error(`Failed to load Phase 5 CV artifact (${p5CvRes.status} ${p5CvRes.statusText})`);
      }
      if (!p5DevRes.ok) {
        throw new Error(`Failed to load Phase 5 dev validation artifact (${p5DevRes.status} ${p5DevRes.statusText})`);
      }
      if (!p5WindowRes.ok) {
        throw new Error(`Failed to load Phase 5 window ablation artifact (${p5WindowRes.status} ${p5WindowRes.statusText})`);
      }
      if (!p5CapRes.ok) {
        throw new Error(`Failed to load Phase 5 target cap artifact (${p5CapRes.status} ${p5CapRes.statusText})`);
      }
      if (!p5HyperRes.ok) {
        throw new Error(`Failed to load Phase 5 hyperparameter search artifact (${p5HyperRes.status} ${p5HyperRes.statusText})`);
      }
      if (!manifestRes.ok) {
        throw new Error(`Failed to load model manifest artifact (${manifestRes.status} ${manifestRes.statusText})`);
      }

      const [
        p4BenchmarkText,
        p5CvText,
        p5DevText,
        p5WindowText,
        p5CapText,
        p5HyperText,
        manifestJson,
      ] = await Promise.all([
        p4BenchmarkRes.text(),
        p5CvRes.text(),
        p5DevRes.text(),
        p5WindowRes.text(),
        p5CapRes.text(),
        p5HyperRes.text(),
        manifestRes.json(),
      ]);

      const phase4Benchmark = parsePhase4BenchmarkCsv(p4BenchmarkText);
      const phase5CrossValidation = parsePhase5CrossValidationCsv(p5CvText);
      const phase5DevelopmentValidation = parsePhase5DevelopmentValidationCsv(p5DevText);
      const phase5WindowAblation = parsePhase5WindowAblationCsv(p5WindowText);
      const phase5TargetCapSensitivity = parsePhase5TargetCapSensitivityCsv(p5CapText);
      const hyperparameterTrials = parseHyperparameterSearchCsv(p5HyperText);
      const manifest = parseModelManifest(manifestJson);

      const result: LoadedModelEvaluationData = {
        phase4Benchmark,
        phase5CrossValidation,
        phase5DevelopmentValidation,
        phase5WindowAblation,
        phase5TargetCapSensitivity,
        hyperparameterTrials,
        manifest,
        syncedTimestamp: manifest.frozen_timestamp,
      };

      cachedData = result;
      return result;
    } finally {
      inflightPromise = null;
    }
  })();

  return inflightPromise;
}

export function clearModelEvaluationCache(): void {
  cachedData = null;
  inflightPromise = null;
}

export function useModelEvaluationData() {
  const [data, setData] = useState<LoadedModelEvaluationData | null>(cachedData);
  const [loading, setLoading] = useState<boolean>(!cachedData);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(() => {
    setLoading(true);
    setError(null);
    loadModelEvaluationData(true)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    let mounted = true;
    if (!cachedData) {
      setLoading(true);
      loadModelEvaluationData()
        .then((res) => {
          if (mounted) {
            setData(res);
            setLoading(false);
          }
        })
        .catch((err) => {
          if (mounted) {
            setError(err instanceof Error ? err.message : String(err));
            setLoading(false);
          }
        });
    } else {
      setData(cachedData);
      setLoading(false);
    }
    return () => {
      mounted = false;
    };
  }, []);

  return { data, loading, error, reload };
}
