/**
 * AeroRUL — Research Artifact Synchronization Script
 *
 * Copies verified Phase 4, Phase 5, and Stage 4 benchmark outputs from reports/ into
 * frontend/public/data/ for runtime consumption by the dashboard and evaluation workbench.
 * Also synchronizes and gzip-compresses the raw NASA C-MAPSS source datasets into
 * frontend/public/data/datasets/ for lazy on-demand client-side streaming and decompression.
 *
 * Strict requirements:
 * 1. Resolved relative to this script file (never CWD).
 * 2. Strict allowlist only — no binaries or extraneous files.
 * 3. Atomic copy/compression with existence and size verification.
 * 4. Fails with non-zero exit code if any artifact is missing or empty.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Project root is two levels up from frontend/scripts/
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');

const SYNC_TARGETS = [
  {
    phase: 'phase4',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'phase4'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'phase4'),
    allowlist: [
      'model_benchmark_metrics.csv',
    ],
  },
  {
    phase: 'phase5',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'phase5'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'phase5'),
    allowlist: [
      'final_test_metrics.csv',
      'final_test_predictions.csv',
      'final_model_manifest.json',
      'cross_validation_metrics.csv',
      'development_validation_metrics.csv',
      'hyperparameter_search_results.csv',
      'window_ablation_metrics.csv',
      'target_cap_sensitivity.csv',
    ],
  },
  {
    phase: 'stage4',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'dashboard_stage4'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'stage4'),
    allowlist: [
      'feature_importance_FD001.csv',
      'feature_importance_FD002.csv',
      'feature_importance_FD003.csv',
      'feature_importance_FD004.csv',
      'sensor_importance_FD001.csv',
      'sensor_importance_FD002.csv',
      'sensor_importance_FD003.csv',
      'sensor_importance_FD004.csv',
      'feature_family_importance.csv',
      'feature_importance_manifest.json',
      'sensor_trajectories_FD001.csv',
      'sensor_trajectories_FD002.csv',
      'sensor_trajectories_FD003.csv',
      'sensor_trajectories_FD004.csv',
      'sensor_trajectories_manifest.json',
    ],
  },
  {
    phase: 'dataset_explorer_metadata',
    sourceDir: path.join(PROJECT_ROOT, 'reports', 'dashboard_dataset_explorer'),
    destDir: path.resolve(__dirname, '..', 'public', 'data', 'datasets'),
    allowlist: [
      'dataset_manifest.json',
      'engine_index_FD001_train.csv',
      'engine_index_FD001_test.csv',
      'engine_index_FD002_train.csv',
      'engine_index_FD002_test.csv',
      'engine_index_FD003_train.csv',
      'engine_index_FD003_test.csv',
      'engine_index_FD004_train.csv',
      'engine_index_FD004_test.csv',
    ],
  },
];

const RAW_DATASETS_CONFIG = {
  sourceDir: path.join(PROJECT_ROOT, 'CMAPSSData'),
  destDir: path.resolve(__dirname, '..', 'public', 'data', 'datasets'),
  filesToCompress: [
    'train_FD001.txt',
    'train_FD002.txt',
    'train_FD003.txt',
    'train_FD004.txt',
    'test_FD001.txt',
    'test_FD002.txt',
    'test_FD003.txt',
    'test_FD004.txt',
  ],
  rulFiles: [
    'RUL_FD001.txt',
    'RUL_FD002.txt',
    'RUL_FD003.txt',
    'RUL_FD004.txt',
  ],
};

console.log('[sync-artifacts] Starting research artifact synchronization...');

let totalSynced = 0;
let totalExpected = 0;

for (const target of SYNC_TARGETS) {
  console.log(`[sync-artifacts] Processing ${target.phase}...`);
  console.log(`  Source: ${target.sourceDir}`);
  console.log(`  Target: ${target.destDir}`);

  if (!fs.existsSync(target.sourceDir)) {
    console.error(`[sync-artifacts] ERROR: Source directory not found: ${target.sourceDir}`);
    process.exit(1);
  }

  if (!fs.existsSync(target.destDir)) {
    fs.mkdirSync(target.destDir, { recursive: true });
    console.log(`  Created directory: ${target.destDir}`);
  }

  for (const filename of target.allowlist) {
    totalExpected++;
    const sourcePath = path.join(target.sourceDir, filename);
    const destPath = path.join(target.destDir, filename);

    if (!fs.existsSync(sourcePath)) {
      console.error(`[sync-artifacts] ERROR: Required artifact missing: ${sourcePath}`);
      process.exit(1);
    }

    const stat = fs.statSync(sourcePath);
    if (stat.size === 0) {
      console.error(`[sync-artifacts] ERROR: Artifact is empty (0 bytes): ${sourcePath}`);
      process.exit(1);
    }

    try {
      fs.copyFileSync(sourcePath, destPath);
      const destStat = fs.statSync(destPath);
      if (destStat.size !== stat.size) {
        throw new Error(`Size mismatch after copy (${destStat.size} vs ${stat.size} bytes)`);
      }
      console.log(`  Synced: ${filename} (${(stat.size / 1024).toFixed(1)} KB)`);
      totalSynced++;
    } catch (err) {
      console.error(`[sync-artifacts] ERROR: Failed to copy ${filename}:`, err);
      process.exit(1);
    }
  }
}

// Synchronize and compress raw C-MAPSS data
console.log('[sync-artifacts] Processing raw C-MAPSS source data compression...');
if (!fs.existsSync(RAW_DATASETS_CONFIG.sourceDir)) {
  console.error(`[sync-artifacts] ERROR: Raw CMAPSSData directory missing: ${RAW_DATASETS_CONFIG.sourceDir}`);
  process.exit(1);
}

if (!fs.existsSync(RAW_DATASETS_CONFIG.destDir)) {
  fs.mkdirSync(RAW_DATASETS_CONFIG.destDir, { recursive: true });
}

for (const filename of RAW_DATASETS_CONFIG.filesToCompress) {
  totalExpected++;
  const sourcePath = path.join(RAW_DATASETS_CONFIG.sourceDir, filename);
  const gzDestPath = path.join(RAW_DATASETS_CONFIG.destDir, `${filename}.gz`);

  if (!fs.existsSync(sourcePath)) {
    console.error(`[sync-artifacts] ERROR: Raw source file missing: ${sourcePath}`);
    process.exit(1);
  }

  const rawBuffer = fs.readFileSync(sourcePath);
  if (rawBuffer.length === 0) {
    console.error(`[sync-artifacts] ERROR: Raw source file is empty: ${sourcePath}`);
    process.exit(1);
  }

  // Compress with max compression level 9
  const compressed = zlib.gzipSync(rawBuffer, { level: 9 });
  
  // Verify byte-for-byte roundtrip integrity
  const roundtrip = zlib.gunzipSync(compressed);
  if (!roundtrip.equals(rawBuffer)) {
    console.error(`[sync-artifacts] ERROR: Gzip roundtrip verification failed for ${filename}`);
    process.exit(1);
  }

  fs.writeFileSync(gzDestPath, compressed);
  const gzStat = fs.statSync(gzDestPath);
  console.log(
    `  Compressed: ${filename} (${(rawBuffer.length / 1024 / 1024).toFixed(2)} MB -> ${(gzStat.size / 1024).toFixed(1)} KB gz)`
  );
  totalSynced++;
}

// Copy official RUL files
for (const filename of RAW_DATASETS_CONFIG.rulFiles) {
  totalExpected++;
  const sourcePath = path.join(RAW_DATASETS_CONFIG.sourceDir, filename);
  const destPath = path.join(RAW_DATASETS_CONFIG.destDir, filename);

  if (!fs.existsSync(sourcePath)) {
    console.error(`[sync-artifacts] ERROR: RUL file missing: ${sourcePath}`);
    process.exit(1);
  }

  const stat = fs.statSync(sourcePath);
  if (stat.size === 0) {
    console.error(`[sync-artifacts] ERROR: RUL file is empty: ${sourcePath}`);
    process.exit(1);
  }

  fs.copyFileSync(sourcePath, destPath);
  console.log(`  Synced: ${filename} (${stat.size} bytes)`);
  totalSynced++;
}

console.log(`[sync-artifacts] Successfully synchronized and compressed ${totalSynced}/${totalExpected} artifacts.`);
