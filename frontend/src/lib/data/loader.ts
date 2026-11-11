/**
 * AeroRUL — Data Loader
 *
 * Fetches and parses research artifacts from the public data directory.
 * Resolves paths using Vite base URL to support subpath hosting.
 * Never fetches from external or mock services.
 */

import {
  TestMetricRecord,
  PredictionRecord,
  ModelManifest,
} from './types';
import {
  parseMetricsCsv,
  parsePredictionsCsv,
  parseModelManifest,
} from './parser';

export interface LoadedBenchmarkData {
  metrics: TestMetricRecord[];
  predictions: PredictionRecord[];
  manifest: ModelManifest;
  syncedTimestamp: string;
}

let cachedData: LoadedBenchmarkData | null = null;
let inflightPromise: Promise<LoadedBenchmarkData> | null = null;

function getArtifactUrl(filename: string): string {
  const base = import.meta.env.BASE_URL || './';
  const cleanBase = base.endsWith('/') ? base : `${base}/`;
  return `${cleanBase}data/phase5/${filename}`;
}

export async function loadBenchmarkData(forceReload = false): Promise<LoadedBenchmarkData> {
  if (cachedData && !forceReload) {
    return cachedData;
  }

  if (inflightPromise && !forceReload) {
    return inflightPromise;
  }

  inflightPromise = (async () => {
    try {
      const metricsUrl = getArtifactUrl('final_test_metrics.csv');
      const predictionsUrl = getArtifactUrl('final_test_predictions.csv');
      const manifestUrl = getArtifactUrl('final_model_manifest.json');

      const [metricsRes, predictionsRes, manifestRes] = await Promise.all([
        fetch(metricsUrl),
        fetch(predictionsUrl),
        fetch(manifestUrl),
      ]);

      if (!metricsRes.ok) {
        throw new Error(`Failed to load metrics artifact (${metricsRes.status} ${metricsRes.statusText}) from ${metricsUrl}`);
      }
      if (!predictionsRes.ok) {
        throw new Error(`Failed to load predictions artifact (${predictionsRes.status} ${predictionsRes.statusText}) from ${predictionsUrl}`);
      }
      if (!manifestRes.ok) {
        throw new Error(`Failed to load manifest artifact (${manifestRes.status} ${manifestRes.statusText}) from ${manifestUrl}`);
      }

      const [metricsText, predictionsText, manifestJson] = await Promise.all([
        metricsRes.text(),
        predictionsRes.text(),
        manifestRes.json(),
      ]);

      const metrics = parseMetricsCsv(metricsText);
      const predictions = parsePredictionsCsv(predictionsText);
      const manifest = parseModelManifest(manifestJson);

      const result: LoadedBenchmarkData = {
        metrics,
        predictions,
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

export function clearBenchmarkCache(): void {
  cachedData = null;
  inflightPromise = null;
}

