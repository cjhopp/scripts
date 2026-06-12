#!/usr/bin/env python3
"""Phase 1 Investigation: Root-cause analysis for missing/zero coarse lags in pairs 625-640.

This script:
1. Regenerates xcorr diagnostic plots for pairs 625-640 with the corrected window asymmetry
2. Dumps published dt, envelope_lag, envelope_peak_cc, xcorr_peak_cc from the bundle
3. Confirms whether 638-640 are NaN-rejected or 625-637 frozen near zero

Output:
- Diagnostic PNG plots in /home/chopp/cassm_local/live/xcorr_diag_plots/
- Console dump of published metrics for pairs 625-640
"""

import sys
import logging
from pathlib import Path
import json

import numpy as np

sys.path.insert(0, '/home/chopp/scripts/python/vm')

from cussp_cassm_diag_xcorr_stages import run_xcorr_diag
from cussp_cassm_process import CASSMTempGather

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONFIG_PATH = "/home/chopp/scripts/python/vm/cussp_cassm_config.yaml"
CACHE_FILE = Path("/home/chopp/cassm_local/live/cassm_tempgather_full.h5")
BUNDLE_FILE = Path("/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz")
DIAG_OUTPUT_DIR = "/home/chopp/cassm_local/live/xcorr_diag_plots"

def main():
    log.info("=" * 80)
    log.info("PHASE 1 INVESTIGATION: Coarse Search Root-Cause Analysis")
    log.info("Pairs 625-640 (DML→TS hydrophones, open fracture region)")
    log.info("=" * 80)
    
    # Step 1: Regenerate diagnostic plots for pairs 625-640 with corrected window asymmetry
    log.info("\n--- STEP 1: Regenerate diagnostic plots (corrected window asymmetry) ---")
    log.info("Running cussp_cassm_diag_xcorr_stages.py for pairs 625-640...")
    
    pair_indices = list(range(625, 641))  # 625-640 inclusive
    log.info(f"Pair indices: {pair_indices}")
    
    try:
        run_xcorr_diag(
            config_path=CONFIG_PATH,
            pair_indices=pair_indices,
            n_debug_plots=20,  # ~20 epochs per pair for detailed trend visualization
            output_dir=DIAG_OUTPUT_DIR,
            explicit_epochs=None,  # Random selection
        )
        log.info(f"✓ Diagnostic plots saved to {DIAG_OUTPUT_DIR}")
    except Exception as e:
        log.error(f"Failed to generate diagnostic plots: {e}", exc_info=True)
        return 1
    
    # Step 2: Dump published metrics from the bundle for pairs 625-640
    log.info("\n--- STEP 2: Inspect published bundle metrics ---")
    
    if not BUNDLE_FILE.exists():
        log.error(f"Bundle file not found: {BUNDLE_FILE}")
        return 1
    
    try:
        bundle = np.load(BUNDLE_FILE, allow_pickle=True)
        log.info(f"Loaded bundle: {list(bundle.keys())}")
        
        # Extract published arrays
        dt_us = bundle.get('dt_us')
        envelope_lag_us = bundle.get('envelope_lag_us')
        envelope_peak_cc = bundle.get('envelope_peak_cc')
        xcorr_peak_cc = bundle.get('xcorr_peak_cc')
        
        if dt_us is None:
            log.error("'dt_us' not found in bundle")
            return 1
        
        log.info(f"dt_us shape: {dt_us.shape}")
        log.info(f"envelope_lag_us shape: {envelope_lag_us.shape if envelope_lag_us is not None else 'None'}")
        
        # Dump per-pair statistics for 625-640
        log.info("\n" + "=" * 100)
        log.info("Published Metrics for Pairs 625-640 (rows: pair, columns: epochs)")
        log.info("=" * 100)
        
        pair_labels = {
            625: "DML(8)→TS(49)", 626: "DML(8)→TS(50)", 627: "DML(8)→TS(51)", 628: "DML(8)→TS(52)",
            629: "DML(8)→TS(53)", 630: "DML(8)→TS(54)", 631: "DML(8)→TS(55)", 632: "DML(8)→TS(56)",
            633: "DML(8)→TS(57)", 634: "DML(8)→TS(58)", 635: "DML(8)→TS(59)", 636: "DML(8)→TS(60)",
            637: "DML(8)→TS(61)", 638: "DML(8)→TS(62)", 639: "DML(8)→TS(63)", 640: "DML(8)→TS(64)",
        }
        
        for p in pair_indices:
            if p >= dt_us.shape[0]:
                log.warning(f"Pair {p} out of range (max {dt_us.shape[0]-1})")
                continue
            
            row_dt = dt_us[p, :]
            row_env_lag = envelope_lag_us[p, :] if envelope_lag_us is not None else None
            row_env_cc = envelope_peak_cc[p, :] if envelope_peak_cc is not None else None
            row_xcorr_cc = xcorr_peak_cc[p, :]
            
            # Statistics
            n_finite_dt = np.isfinite(row_dt).sum()
            n_total = len(row_dt)
            n_nan_dt = n_total - n_finite_dt
            
            if n_finite_dt > 0:
                dt_valid = row_dt[np.isfinite(row_dt)]
                dt_mean = np.mean(dt_valid)
                dt_std = np.std(dt_valid)
                dt_min = np.min(dt_valid)
                dt_max = np.max(dt_valid)
            else:
                dt_mean = dt_std = dt_min = dt_max = np.nan
            
            # Envelope coarse lag stats
            if row_env_lag is not None:
                env_lag_valid = row_env_lag[np.isfinite(row_env_lag)]
                env_lag_mean = np.mean(env_lag_valid) if len(env_lag_valid) > 0 else np.nan
            else:
                env_lag_mean = np.nan
            
            # Envelope cc stats
            if row_env_cc is not None:
                env_cc_mean = np.mean(row_env_cc[np.isfinite(row_env_cc)]) if np.any(np.isfinite(row_env_cc)) else np.nan
            else:
                env_cc_mean = np.nan
            
            # Xcorr fine cc stats
            xcorr_cc_valid = row_xcorr_cc[np.isfinite(row_xcorr_cc)]
            xcorr_cc_mean = np.mean(xcorr_cc_valid) if len(xcorr_cc_valid) > 0 else np.nan
            
            # Status indicator
            if n_nan_dt == n_total:
                status = "ALL_NAN_REJECTED"
            elif n_nan_dt > n_total * 0.5:
                status = "MOSTLY_REJECTED"
            elif np.abs(dt_mean) < 10.0 and dt_max < 20.0:
                status = "FROZEN_NEAR_ZERO"
            else:
                status = "OK"
            
            log.info(
                f"Pair {p:3d} {pair_labels.get(p, ''):20s} | "
                f"finite:{n_finite_dt:4d}/{n_total:4d} "
                f"dt_us(mean±std):{dt_mean:8.1f}±{dt_std:6.1f} "
                f"[{dt_min:8.1f},{dt_max:8.1f}] "
                f"env_lag_mean:{env_lag_mean:8.1f}µs "
                f"env_cc:{env_cc_mean:5.2f} "
                f"xcorr_cc:{xcorr_cc_mean:5.2f} | "
                f"Status: {status}"
            )
        
        log.info("=" * 100)
        
        # Root-cause hypothesis check
        log.info("\n--- ROOT-CAUSE HYPOTHESIS ---")
        row_638 = dt_us[638, :]
        row_639 = dt_us[639, :]
        row_640 = dt_us[640, :]
        
        nan_count_638 = (~np.isfinite(row_638)).sum()
        nan_count_639 = (~np.isfinite(row_639)).sum()
        nan_count_640 = (~np.isfinite(row_640)).sum()
        
        if nan_count_638 > len(row_638) * 0.5:
            log.info("✗ HYPOTHESIS 1 CONFIRMED: Pair 638 mostly NaN-rejected (envelope_min_peak_cc gate)")
        else:
            log.info("○ Pair 638 not mostly NaN; check envelope_peak_cc gate threshold")
        
        if nan_count_639 > len(row_639) * 0.5:
            log.info("✗ HYPOTHESIS 1 CONFIRMED: Pair 639 mostly NaN-rejected")
        else:
            log.info("○ Pair 639 not mostly NaN")
        
        if nan_count_640 > len(row_640) * 0.5:
            log.info("✗ HYPOTHESIS 1 CONFIRMED: Pair 640 mostly NaN-rejected")
        else:
            log.info("○ Pair 640 not mostly NaN")
        
        # Check 625-637 for frozen-near-zero pattern
        dt_625_637 = [np.mean(dt_us[p, np.isfinite(dt_us[p, :])]) for p in range(625, 638)]
        if all(np.abs(dt) < 15.0 for dt in dt_625_637 if np.isfinite(dt)):
            log.info("✗ HYPOTHESIS 2 CONFIRMED: Pairs 625-637 all frozen near zero (<15 µs magnitude)")
        else:
            log.info(f"○ Pairs 625-637 show variation: {[f'{dt:.1f}' for dt in dt_625_637]}")
        
        # Log guide-smoothing effect hint
        log.info("\nDiagnostic plots show per-pair guide-lag smoothing state across all epochs.")
        log.info("If guide-median freezes on low env_cc (not updated), fine xcorr centers on stale ~0 lag.")
        log.info("Compare Panel 6 (smoothed_lag timeseries) against coarse_lag: frozen = root cause.")
        
    except Exception as e:
        log.error(f"Failed to inspect bundle: {e}", exc_info=True)
        return 1
    
    log.info("\n" + "=" * 80)
    log.info("PHASE 1 INVESTIGATION COMPLETE")
    log.info("Next: Review diagnostic plots and confirm root cause before Phase 2 fix.")
    log.info("=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
