#!/usr/bin/env python3
"""Diagnostic: inspect envelope vs fine-xcorr dt for pairs 625-646."""
import numpy as np

BUNDLE = "/home/chopp/cassm_local/live/cassm_dashboard_bundle_full.npz"

b = np.load(BUNDLE, allow_pickle=True)
print("Keys:", b.files)

n_src = int(b["n_sources"])
n_rec = int(b["n_receivers"])
print(f"n_sources={n_src}, n_receivers={n_rec}, n_pairs={n_src*n_rec}")

dt      = b["dt_us"].astype(np.float32)
cc      = b["xcorr_peak_cc"].astype(np.float32) if "xcorr_peak_cc" in b.files else None
env_cc  = b["envelope_peak_cc"].astype(np.float32) if "envelope_peak_cc" in b.files else None
env_lag = b["envelope_lag_us"].astype(np.float32) if "envelope_lag_us" in b.files else None
rms     = b["rms"].astype(np.float32)

print(f"\ndt shape:       {dt.shape}")
print(f"env_cc present: {env_cc is not None}")
print(f"env_lag present:{env_lag is not None}")

# Global fraction of epochs where env_lag is finite (non-NaN)
if env_lag is not None:
    frac_env_finite = float(np.mean(np.isfinite(env_lag)))
    print(f"\nGlobal env_lag finite fraction: {frac_env_finite:.3f}")
    # Check: are all env_lag NaN? (would mean old bundle without envelope mode)
    print(f"All env_lag NaN? {np.all(~np.isfinite(env_lag))}")
    print(f"env_lag unique non-nan count: {np.sum(np.isfinite(env_lag))}")
    # Sample a few values from early pairs
    print(f"env_lag[0, :5] = {env_lag[0, :5]}")
    print(f"env_lag[625, :5] = {env_lag[625, :5]}")

print()
print(f"{'p':>5} {'rms_nz':>6} {'fin_dt':>6} | {'dt_med':>8} {'env_lag_med':>12} | "
      f"{'cc_med':>6} {'env_cc_med':>10} | {'env_fin':>7} {'env_but_nan':>11} | "
      f"{'discr_max':>10} {'discr_med':>10}")
print("-" * 110)

for p in range(625, 647):
    dt_p     = dt[p, :]
    rms_p    = rms[p, :]
    cc_p     = cc[p, :] if cc is not None else None
    env_cc_p = env_cc[p, :] if env_cc is not None else None
    env_lag_p= env_lag[p, :] if env_lag is not None else None

    n_rms_nz  = int(np.sum(rms_p > 0))
    n_finite_dt = int(np.sum(np.isfinite(dt_p)))
    dt_med    = float(np.nanmedian(dt_p)) if n_finite_dt else float('nan')
    cc_med    = float(np.nanmedian(cc_p)) if cc_p is not None else float('nan')

    if env_lag_p is not None:
        env_finite    = np.isfinite(env_lag_p)
        n_env_finite  = int(np.sum(env_finite))
        env_lag_med   = float(np.nanmedian(env_lag_p)) if n_env_finite else float('nan')
        n_env_but_nan = int(np.sum(env_finite & ~np.isfinite(dt_p)))
        env_cc_med    = float(np.nanmedian(env_cc_p)) if env_cc_p is not None else float('nan')
        both = env_finite & np.isfinite(dt_p)
        if np.any(both):
            diff = dt_p[both] - env_lag_p[both]
            max_discr = float(np.max(np.abs(diff)))
            med_discr = float(np.median(np.abs(diff)))
        else:
            max_discr = med_discr = float('nan')
    else:
        n_env_finite = n_env_but_nan = -1
        env_lag_med = env_cc_med = max_discr = med_discr = float('nan')

    print(f"{p:5d} {n_rms_nz:6d} {n_finite_dt:6d} | {dt_med:+8.1f} {env_lag_med:+12.1f} | "
          f"{cc_med:6.3f} {env_cc_med:10.3f} | {n_env_finite:7d} {n_env_but_nan:11d} | "
          f"{max_discr:10.1f} {med_discr:10.1f}")

# Also check: did this bundle come from a run with envelope mode actually on?
# If env_lag is all NaN it means the old cached metrics were served.
print()
print("=== accept_max_lag check for p=625 ===")
p = 625
if cc is not None and env_lag is not None:
    cc_p = cc[p, :]
    dt_p = dt[p, :]
    env_lag_p = env_lag[p, :]
    env_cc_p = env_cc[p, :]
    both = np.isfinite(env_lag_p) & np.isfinite(dt_p)
    if np.any(both):
        diff = dt_p[both] - env_lag_p[both]
        print(f"  epochs with both finite: {np.sum(both)}")
        print(f"  dt - env_lag percentiles [5,25,50,75,95]: "
              f"{np.percentile(diff, [5,25,50,75,95])}")
    else:
        print("  No epochs with both dt and env_lag finite")
    print(f"  env_cc_p[:10] = {env_cc_p[:10]}")
    print(f"  env_lag_p[:10] = {env_lag_p[:10]}")
    print(f"  dt_p[:10] = {dt_p[:10]}")
    print(f"  cc_p[:10] = {cc_p[:10]}")
