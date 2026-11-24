/**
 * AeroRUL — Dataset Explorer Loader & Cache
 *
 * Handles lazy on-demand streaming, gzip decompression, parsing,
 * and memory caching of raw C-MAPSS observations and dataset metadata.
 */

import {
  DatasetManifest,
  DatasetSplit,
  EngineIndexRecord,
  RawObservationRecord,
} from './datasetExplorerTypes';
import {
  parseEngineIndex,
  parseRawObservationText,
  parseRulVectorText,
} from './datasetExplorerParser';
import { DatasetId } from './types';

// In-memory caches to prevent redundant network fetches and re-parsing
let cachedManifest: DatasetManifest | null = null;
const engineIndexCache = new Map<string, EngineIndexRecord[]>();
const rawObservationsCache = new Map<string, RawObservationRecord[]>();
const rulVectorCache = new Map<string, number[]>();

/**
 * Clear all in-memory caches (useful for test resets).
 */
export function clearDatasetExplorerCache(): void {
  cachedManifest = null;
  engineIndexCache.clear();
  rawObservationsCache.clear();
  rulVectorCache.clear();
}

/**
 * Fetch and cache dataset manifest JSON.
 */
export async function loadDatasetManifest(basePath = '/data/datasets'): Promise<DatasetManifest> {
  if (cachedManifest) return cachedManifest;

  const url = `${basePath}/dataset_manifest.json`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load dataset manifest from ${url} (HTTP ${res.status})`);
  }

  const manifest = (await res.json()) as DatasetManifest;
  cachedManifest = manifest;
  return manifest;
}

/**
 * Fetch, parse, and cache engine index for a specific dataset and split.
 */
export async function loadEngineIndex(
  dataset: DatasetId,
  split: DatasetSplit,
  basePath = '/data/datasets'
): Promise<EngineIndexRecord[]> {
  const cacheKey = `${dataset}_${split}`;
  if (engineIndexCache.has(cacheKey)) {
    return engineIndexCache.get(cacheKey)!;
  }

  const url = `${basePath}/engine_index_${dataset}_${split}.csv`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load engine index from ${url} (HTTP ${res.status})`);
  }

  const csvText = await res.text();
  const records = parseEngineIndex(csvText);
  engineIndexCache.set(cacheKey, records);
  return records;
}

/**
 * Decompress gzip ArrayBuffer or Response to text.
 * Safely checks gzip magic header (0x1f, 0x8b) to handle both raw gzip binary
 * and transparently decompressed/plaintext responses.
 */
async function decompressGzip(res: Response): Promise<string> {
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);

  const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;

  if (isGzip && typeof DecompressionStream !== 'undefined') {
    try {
      const ds = new DecompressionStream('gzip');
      const decompressedStream = new Response(bytes).body!.pipeThrough(ds);
      return await new Response(decompressedStream).text();
    } catch (err) {
      console.warn('DecompressionStream error on gzip buffer:', err);
    }
  }

  return new TextDecoder().decode(bytes);
}

/**
 * Lazily load, decompress, parse, and cache raw C-MAPSS observations.
 * Fetches only the requested (dataset, split) file.
 * Verifies that the parsed record count matches the dataset manifest's expected row count
 * before caching the split. Partial or corrupted datasets are never cached.
 */
export async function loadRawObservations(
  dataset: DatasetId,
  split: DatasetSplit,
  basePath = '/data/datasets',
  expectedRowCount?: number
): Promise<RawObservationRecord[]> {
  const cacheKey = `${dataset}_${split}`;
  if (rawObservationsCache.has(cacheKey)) {
    return rawObservationsCache.get(cacheKey)!;
  }

  const filename = `${split}_${dataset}.txt.gz`;
  const url = `${basePath}/${filename}`;
  const res = await fetch(url);

  if (!res.ok) {
    throw new Error(`Failed to fetch raw dataset ${filename} from ${url} (HTTP ${res.status})`);
  }

  const text = await decompressGzip(res);
  const records = parseRawObservationText(text);

  // Manifest row-count verification before caching
  let expected = expectedRowCount;
  if (expected == null) {
    try {
      const manifest = await loadDatasetManifest(basePath);
      expected = manifest.datasets[dataset]?.[split]?.row_count;
    } catch {
      // If manifest is unavailable, skip manifest row-count verification
    }
  }

  if (typeof expected === 'number' && records.length !== expected) {
    throw new Error(
      `Dataset integrity violation: parsed ${records.length} records for ${dataset} (${split}), but manifest expects ${expected} rows. Partial dataset not cached.`
    );
  }

  rawObservationsCache.set(cacheKey, records);
  return records;
}

/**
 * Load official test RUL ground-truth vector for a dataset.
 */
export async function loadRulVector(
  dataset: DatasetId,
  basePath = '/data/datasets'
): Promise<number[]> {
  const cacheKey = dataset;
  if (rulVectorCache.has(cacheKey)) {
    return rulVectorCache.get(cacheKey)!;
  }

  const url = `${basePath}/RUL_${dataset}.txt`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load RUL file from ${url} (HTTP ${res.status})`);
  }

  const text = await res.text();
  const vector = parseRulVectorText(text);
  rulVectorCache.set(cacheKey, vector);
  return vector;
}
