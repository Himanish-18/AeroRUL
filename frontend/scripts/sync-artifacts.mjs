/**
 * AeroRUL — Research Artifact Synchronization Script
 *
 * Copies verified Phase 5 benchmark outputs from reports/phase5/ into
 * frontend/public/data/phase5/ for runtime consumption by the dashboard.
 *
 * Strict requirements:
 * 1. Resolved relative to this script file (never CWD).
 * 2. Strict allowlist only — no binaries or extraneous files.
 * 3. Atomic copy with existence and size verification.
 * 4. Fails with non-zero exit code if any artifact is missing or empty.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Project root is two levels up from frontend/scripts/
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const SOURCE_DIR = path.join(PROJECT_ROOT, 'reports', 'phase5');
const DEST_DIR = path.resolve(__dirname, '..', 'public', 'data', 'phase5');

const ARTIFACT_ALLOWLIST = [
  'final_test_metrics.csv',
  'final_test_predictions.csv',
  'final_model_manifest.json'
];

console.log('[sync-artifacts] Starting research artifact synchronization...');
console.log(`[sync-artifacts] Source root: ${SOURCE_DIR}`);
console.log(`[sync-artifacts] Target root: ${DEST_DIR}`);

// 1. Verify source directory exists
if (!fs.existsSync(SOURCE_DIR)) {
  console.error(`[sync-artifacts] ERROR: Source directory not found: ${SOURCE_DIR}`);
  process.exit(1);
}

// 2. Ensure target directory exists
if (!fs.existsSync(DEST_DIR)) {
  fs.mkdirSync(DEST_DIR, { recursive: true });
  console.log(`[sync-artifacts] Created target directory: ${DEST_DIR}`);
}

let syncCount = 0;

for (const filename of ARTIFACT_ALLOWLIST) {
  const sourcePath = path.join(SOURCE_DIR, filename);
  const destPath = path.join(DEST_DIR, filename);

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
    console.log(`[sync-artifacts] Synced: ${filename} (${(stat.size / 1024).toFixed(1)} KB)`);
    syncCount++;
  } catch (err) {
    console.error(`[sync-artifacts] ERROR: Failed to copy ${filename}:`, err);
    process.exit(1);
  }
}

console.log(`[sync-artifacts] Successfully synchronized ${syncCount}/${ARTIFACT_ALLOWLIST.length} artifacts.`);

