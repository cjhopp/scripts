#!/usr/bin/env python3
"""CDD-TLFWI engine for CUSSP CASSM — Phase 1: hybrid FWI dt estimator.

This module provides a cycle-skip-resistant travel-time estimator for
DM*→TS hydrophone pairs using a model-based approach:

  1. Build a 2D velocity model v_p(x,z) for the DM-borehole → TS-string
     vertical cross-section via FATT (straight-ray back-projection from
     existing AIC baseline picks).
  2. Estimate a per-source wavelet by windowing the baseline first arrival
     and stacking over receivers.
  3. For each epoch-pair, run a 1D line search over time shift dt that
     minimizes the correlative misfit (1 - NCC) between the forward-modelled
     synthetic and the observed waveform.  This is iterated coarse-to-fine
     across a multiscale frequency band progression, mirroring the CDD-TLFWI
     strategy from:
       Liu et al. (2022) "Correlative Double-Difference Time-Lapse FWI"

The entry point called from compute_metrics() is fwi_estimate_dt(), which
returns (dt_us, peak_ncc, rejected) — the same signature as
_xcorr_dt_samples() so no structural changes are required in the caller.

Phase 2 (full δv_p spatial inversion) stubs are included at the end of
this file but are not called from the current processing pipeline.

Dependencies: numpy, scipy (always available in the CUSSP environment).
Optional: devito (for DevitoSolver backend).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

LOG = logging.getLogger("cussp_cassm_fwi")

# ---------------------------------------------------------------------------
# FWIGrid — 2D vertical cross-section grid
# ---------------------------------------------------------------------------

@dataclass
class FWIGrid:
    """2D vertical cross-section grid for the FD wave equation solver.

    The grid covers the vertical plane containing the DM source borehole(s)
    and the TS hydrophone string.  Horizontal axis x runs along the
    inter-borehole direction; vertical axis z runs downward (positive = deeper).
    All coordinate values are in metres, consistent with the HMC frame used
    by cussp_cassm_inversion_prep.py.
    """
    nx: int          # number of horizontal grid points
    nz: int          # number of vertical (depth) grid points
    dx: float        # horizontal grid spacing (m)
    dz: float        # vertical grid spacing (m)
    x0: float        # origin x (HMC easting, m)
    z0: float        # origin z (HMC depth, m; z increases downward in grid index)
    dt: float        # time step (s) — must satisfy CFL condition
    nt: int          # number of time steps
    x: np.ndarray    # shape (nx,) — x coordinates in metres
    z: np.ndarray    # shape (nz,) — z coordinates in metres (increasing downward)

    @classmethod
    def from_source_receiver_positions(
        cls,
        src_xyz: np.ndarray,          # (n_src, 3) HMC x,y,z in metres
        rec_xyz: np.ndarray,          # (n_rec, 3) HMC x,y,z in metres
        dx: float = 0.5,
        dz: float = 0.5,
        dt: Optional[float] = None,   # None → auto CFL from vp_max_estimate
        vp_max_estimate: float = 5000.0,
        record_time_s: float = 0.100, # simulation duration (s); covers full trace
        padding_m: float = 20.0,
    ) -> "FWIGrid":
        """Construct a 2D grid that spans all source and receiver positions.

        The 2D plane is defined as the vertical cross-section whose horizontal
        axis aligns with the projection of the source-to-receiver inter-borehole
        vector onto the horizontal plane.  All source and receiver positions are
        projected onto this plane (their out-of-plane component is discarded).

        Parameters
        ----------
        src_xyz, rec_xyz : (N, 3) arrays of HMC coordinates (easting, northing, depth)
        dx, dz           : grid spacings in metres
        dt               : time step in seconds; auto-computed from CFL if None
        vp_max_estimate  : estimated maximum v_p (m/s) for CFL and record length
        record_time_s    : simulation window length (s) — set >= max expected traveltime
        padding_m        : extra grid margin beyond source/receiver extent (m)
        """
        all_xyz = np.vstack([src_xyz, rec_xyz])

        # Horizontal inter-borehole axis: mean source position → mean receiver position
        src_mean = src_xyz[:, :2].mean(axis=0)   # (easting, northing)
        rec_mean = rec_xyz[:, :2].mean(axis=0)
        axis = rec_mean - src_mean
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-3:
            # Degenerate (src and rec at same horizontal position) — use easting axis
            axis_unit = np.array([1.0, 0.0])
        else:
            axis_unit = axis / axis_len

        # Project all positions onto (horizontal_along_axis, depth) plane
        horiz = all_xyz[:, :2].dot(axis_unit)   # projection along inter-borehole axis
        depth = -all_xyz[:, 2]                  # HMC z is negative-downward; flip so grid z≥0

        x_min = horiz.min() - padding_m
        x_max = horiz.max() + padding_m
        z_min = max(depth.min() - padding_m, 0.0)
        z_max = depth.max() + padding_m

        nx = max(int(math.ceil((x_max - x_min) / dx)) + 1, 4)
        nz = max(int(math.ceil((z_max - z_min) / dz)) + 1, 4)

        # CFL condition: dt < dx / (sqrt(2) * vp_max)
        cfl_dt = 0.9 * min(dx, dz) / (math.sqrt(2.0) * vp_max_estimate)
        if dt is None:
            dt = cfl_dt
        elif dt > cfl_dt:
            LOG.warning(
                "Supplied dt=%.2e s exceeds CFL limit %.2e s; clamping to CFL.",
                dt, cfl_dt,
            )
            dt = cfl_dt

        nt = max(int(math.ceil(record_time_s / dt)), 10)

        x_arr = np.arange(nx, dtype=np.float64) * dx + x_min
        z_arr = np.arange(nz, dtype=np.float64) * dz + z_min

        return cls(
            nx=nx, nz=nz, dx=dx, dz=dz,
            x0=x_min, z0=z_min,
            dt=dt, nt=nt,
            x=x_arr, z=z_arr,
        )

    def world_to_grid(self, x_world: float, z_world: float) -> Tuple[int, int]:
        """Convert world (x, z) coordinates to nearest (ix, iz) grid indices."""
        ix = int(round((x_world - self.x0) / self.dx))
        iz = int(round((z_world - self.z0) / self.dz))
        ix = int(np.clip(ix, 0, self.nx - 1))
        iz = int(np.clip(iz, 0, self.nz - 1))
        return ix, iz

    def project_positions(
        self,
        xyz: np.ndarray,             # (N, 3) HMC x,y,z
        axis_unit: np.ndarray,       # (2,) horizontal unit vector
    ) -> np.ndarray:
        """Project HMC 3D positions onto the 2D grid, return (N, 2) grid indices (ix, iz)."""
        horiz = xyz[:, :2].dot(axis_unit)
        depth = -xyz[:, 2]  # flip sign: HMC z negative-down → grid z positive-down
        ix = np.clip(np.round((horiz - self.x0) / self.dx).astype(int), 0, self.nx - 1)
        iz = np.clip(np.round((depth - self.z0) / self.dz).astype(int), 0, self.nz - 1)
        return np.stack([ix, iz], axis=1)  # (N, 2)


# ---------------------------------------------------------------------------
# SolverBackend protocol
# ---------------------------------------------------------------------------

class SolverBackend(Protocol):
    """Interface that all wave-equation solver backends must implement."""

    def forward(
        self,
        vp: np.ndarray,                    # (nz, nx) velocity model in m/s
        source_wavelet: np.ndarray,        # (nt,) normalized source time function
        src_ix: int,                       # source grid x-index
        src_iz: int,                       # source grid z-index
        rec_ix: np.ndarray,                # (n_rec,) receiver grid x-indices
        rec_iz: np.ndarray,                # (n_rec,) receiver grid z-indices
        grid: FWIGrid,
    ) -> np.ndarray:                       # (n_rec, nt) synthetic seismograms
        ...


# ---------------------------------------------------------------------------
# AnalyticSolver — straight-ray + Ricker wavelet (testing / no-FD fallback)
# ---------------------------------------------------------------------------

class AnalyticSolver:
    """Analytic 1D Green's function solver for pipeline testing.

    Uses straight-ray traveltimes and a Ricker wavelet with 1/r amplitude
    decay.  Completely ignores spatial velocity structure (uses only the
    mean v_p).  Produces physically approximate synthetics suitable for
    verifying the dt line-search and multiscale machinery without running
    a real wave equation.
    """

    def forward(
        self,
        vp: np.ndarray,
        source_wavelet: np.ndarray,
        src_ix: int,
        src_iz: int,
        rec_ix: np.ndarray,
        rec_iz: np.ndarray,
        grid: FWIGrid,
    ) -> np.ndarray:
        vp_mean = float(np.mean(vp[vp > 0])) if np.any(vp > 0) else 3000.0
        nt = grid.nt
        n_rec = len(rec_ix)
        synthetics = np.zeros((n_rec, nt), dtype=np.float32)
        t_arr = np.arange(nt, dtype=np.float64) * grid.dt

        for i in range(n_rec):
            dx_m = (rec_ix[i] - src_ix) * grid.dx
            dz_m = (rec_iz[i] - src_iz) * grid.dz
            dist = max(math.sqrt(dx_m ** 2 + dz_m ** 2), grid.dx)
            t_travel = dist / vp_mean
            amp = 1.0 / dist
            # Shift source_wavelet by travel time via fractional-sample sinc interpolation
            n_shift = t_travel / grid.dt
            n_int = int(math.floor(n_shift))
            frac = n_shift - n_int
            wavelet_padded = np.zeros(nt + n_int + 2, dtype=np.float64)
            wavelet_padded[n_int: n_int + len(source_wavelet)] = source_wavelet
            # Linear interpolation for fractional shift (adequate for analytic mode)
            shifted = (1.0 - frac) * wavelet_padded[:nt] + frac * wavelet_padded[1: nt + 1]
            synthetics[i] = (shifted * amp).astype(np.float32)

        return synthetics


# ---------------------------------------------------------------------------
# FD2DSolver — O(2,2) explicit finite-difference acoustic wave equation
# ---------------------------------------------------------------------------

class FD2DSolver:
    """2D acoustic finite-difference wave equation solver.

    Discretises: ∂²u/∂t² = v_p²(x,z) · [∂²u/∂x² + ∂²u/∂z²] + s(t)·δ(x-xs,z-zs)

    Scheme: O(2,2) in time and space (2nd order in both).
    Boundary conditions: CPML absorbing layers on all four edges.
    Source injection: additive Ricker/arbitrary wavelet at a single grid point.
    Receiver extraction: waveform samples at arbitrary grid points.

    Grid convention: vp shape is (nz, nx), index [iz, ix].
    """

    def __init__(self, cpml_thickness: int = 20, cpml_alpha_max: float = 0.05):
        """
        Parameters
        ----------
        cpml_thickness : number of CPML absorbing layers on each edge
        cpml_alpha_max : maximum CPML damping coefficient (frequency-independent part)
        """
        self.cpml_n = cpml_thickness
        self.cpml_alpha = cpml_alpha_max

    # ------------------------------------------------------------------
    # CPML coefficient construction
    # ------------------------------------------------------------------

    def _cpml_profile(
        self, n_total: int, d: float, vp_max: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sigma, kappa) damping profiles of length n_total.

        sigma and kappa are zero in the interior and ramp smoothly at both edges.
        """
        n = self.cpml_n
        sigma = np.zeros(n_total, dtype=np.float64)
        kappa = np.ones(n_total, dtype=np.float64)
        if n <= 0 or n * 2 >= n_total:
            return sigma, kappa

        sigma_max = (3.0 * vp_max * math.log(1000.0)) / (2.0 * n * d)
        for i in range(n):
            frac = (n - i - 0.5) / n
            val = sigma_max * (frac ** 2)
            sigma[i] = val
            sigma[n_total - 1 - i] = val
            kappa_val = 1.0 + (sigma_max - val) / sigma_max if sigma_max > 0 else 1.0
            kappa[i] = kappa_val
            kappa[n_total - 1 - i] = kappa_val

        return sigma, kappa

    # ------------------------------------------------------------------
    # Forward modelling
    # ------------------------------------------------------------------

    def forward(
        self,
        vp: np.ndarray,
        source_wavelet: np.ndarray,
        src_ix: int,
        src_iz: int,
        rec_ix: np.ndarray,
        rec_iz: np.ndarray,
        grid: FWIGrid,
    ) -> np.ndarray:
        """Simulate acoustic wavefield and extract seismograms at receivers.

        Parameters
        ----------
        vp            : (nz, nx) velocity model in m/s (float64 or float32)
        source_wavelet: (nt,) or shorter source time function; zero-padded to nt
        src_ix, src_iz: integer grid indices of the source
        rec_ix, rec_iz: (n_rec,) integer grid indices of receivers
        grid          : FWIGrid with dt, dx, dz, nt, nz, nx

        Returns
        -------
        synthetics : (n_rec, nt) float32 array of recorded waveforms
        """
        nz, nx = grid.nz, grid.nx
        nt = grid.nt
        dt = grid.dt
        dx = grid.dx
        dz = grid.dz
        n_rec = len(rec_ix)

        vp2 = np.asarray(vp, dtype=np.float64) ** 2  # v_p² for speed

        # CPML profiles
        vp_max = float(np.max(vp2) ** 0.5)
        sigma_x, kappa_x = self._cpml_profile(nx, dx, vp_max)
        sigma_z, kappa_z = self._cpml_profile(nz, dz, vp_max)
        # Memory variables for split-field CPML (1st-order PML formulation)
        psi_x = np.zeros((nz, nx), dtype=np.float64)  # x-direction memory
        psi_z = np.zeros((nz, nx), dtype=np.float64)  # z-direction memory
        b_x = np.exp(-sigma_x * dt)
        b_z = np.exp(-sigma_z * dt)

        # Wavefield arrays (two time levels)
        u_prev = np.zeros((nz, nx), dtype=np.float64)
        u_curr = np.zeros((nz, nx), dtype=np.float64)

        # Prepare source wavelet (zero-pad or truncate to nt)
        src = np.zeros(nt, dtype=np.float64)
        n_wav = min(len(source_wavelet), nt)
        src[:n_wav] = source_wavelet[:n_wav]

        # Clamp source/receiver indices
        src_ix = int(np.clip(src_ix, 0, nx - 1))
        src_iz = int(np.clip(src_iz, 0, nz - 1))
        rec_ix = np.clip(rec_ix, 0, nx - 1).astype(int)
        rec_iz = np.clip(rec_iz, 0, nz - 1).astype(int)

        synthetics = np.zeros((n_rec, nt), dtype=np.float32)

        # dt² precomputed
        dt2 = dt ** 2

        for it in range(nt):
            # --- Spatial derivatives with CPML ---
            # ∂u/∂x using central differences (interior)
            du_dx = np.zeros((nz, nx), dtype=np.float64)
            du_dz = np.zeros((nz, nx), dtype=np.float64)
            du_dx[:, 1:nx - 1] = (u_curr[:, 2:] - u_curr[:, :nx - 2]) / (2.0 * dx)
            du_dz[1:nz - 1, :] = (u_curr[2:, :] - u_curr[:nz - 2, :]) / (2.0 * dz)

            # Update CPML memory variables
            psi_x = b_x[np.newaxis, :] * psi_x + (1.0 - b_x[np.newaxis, :]) * du_dx
            psi_z = b_z[:, np.newaxis] * psi_z + (1.0 - b_z[:, np.newaxis]) * du_dz

            # Modified derivatives
            du_dx_m = du_dx / kappa_x[np.newaxis, :] + psi_x
            du_dz_m = du_dz / kappa_z[:, np.newaxis] + psi_z

            # Second derivatives
            d2u_dx2 = np.zeros((nz, nx), dtype=np.float64)
            d2u_dz2 = np.zeros((nz, nx), dtype=np.float64)
            d2u_dx2[:, 1:nx - 1] = (
                du_dx_m[:, 2:] - du_dx_m[:, :nx - 2]
            ) / (2.0 * dx)
            d2u_dz2[1:nz - 1, :] = (
                du_dz_m[2:, :] - du_dz_m[:nz - 2, :]
            ) / (2.0 * dz)

            laplacian = d2u_dx2 + d2u_dz2

            # Time update: u_next = 2*u_curr - u_prev + dt² * v_p² * laplacian
            u_next = 2.0 * u_curr - u_prev + dt2 * vp2 * laplacian

            # Inject source (additive, normalised by dt²)
            u_next[src_iz, src_ix] += dt2 * vp2[src_iz, src_ix] * src[it]

            # Advance time levels
            u_prev = u_curr
            u_curr = u_next

            # Extract at receiver positions
            for r in range(n_rec):
                synthetics[r, it] = float(u_curr[rec_iz[r], rec_ix[r]])

        return synthetics


# ---------------------------------------------------------------------------
# Utility: bandpass filter (reuses scipy if available)
# ---------------------------------------------------------------------------

def _bandpass(
    x: np.ndarray,
    f_low: float,
    f_high: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Falls back to the raw signal if scipy is unavailable or parameters
    are out of range.
    """
    if not _SCIPY_AVAILABLE:
        return x
    nyq = 0.5 * sample_rate_hz
    lo = f_low / nyq if f_low > 0 else None
    hi = f_high / nyq if f_high > 0 else None
    if lo is not None and lo >= 1.0:
        lo = None
    if hi is not None and hi >= 1.0:
        hi = 0.999
    if lo is None and hi is None:
        return x
    if lo is not None and hi is not None and lo >= hi:
        return x
    if x.size < 8:
        return x
    try:
        if lo is not None and hi is not None:
            sos = butter(order, [lo, hi], btype="band", output="sos")
        elif lo is not None:
            sos = butter(order, lo, btype="high", output="sos")
        else:
            sos = butter(order, hi, btype="low", output="sos")
        return sosfiltfilt(sos, x).astype(x.dtype)
    except Exception:
        return x


# ---------------------------------------------------------------------------
# Ricker wavelet
# ---------------------------------------------------------------------------

def _ricker_wavelet(f_peak: float, dt: float, nt: int, delay_s: float = 0.0) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    Parameters
    ----------
    f_peak  : peak frequency (Hz)
    dt      : time step (s)
    nt      : number of samples
    delay_s : time delay from t=0 to wavelet peak (s); defaults to 0

    Returns
    -------
    wavelet : (nt,) float64 array normalized to unit peak amplitude
    """
    t = np.arange(nt, dtype=np.float64) * dt - delay_s
    pi2f2t2 = (math.pi * f_peak * t) ** 2
    w = (1.0 - 2.0 * pi2f2t2) * np.exp(-pi2f2t2)
    peak = np.max(np.abs(w))
    if peak > 0:
        w /= peak
    return w


# ---------------------------------------------------------------------------
# Source wavelet estimation from baseline data
# ---------------------------------------------------------------------------

def estimate_source_wavelet(
    d_obs_baseline: np.ndarray,   # (n_pairs, nt) observed baseline waveforms
    baseline_picks: np.ndarray,   # (n_pairs,) sample indices from _baseline_picks()
    gate_pre_samples: int,
    gate_post_samples: int,
    src_indices: np.ndarray,      # (n_dm_sources,) source indices (0-based, subset of DM*)
    rec_indices: np.ndarray,      # (n_ts_receivers,) receiver indices (0-based)
    n_receivers: int,
) -> np.ndarray:
    """Estimate per-DM-source wavelet from baseline data.

    Per the paper: "the source wavelet is estimated by windowing out
    first-arrival direct waves and summing waveforms of different traces."

    For each DM source, we window the P-wave arrival on all active TS
    hydrophone receiver traces, stack them, and normalise to unit amplitude.
    This stacking suppresses noise and emphasises the source signature.

    Returns
    -------
    wavelets : (n_dm_sources, nt) float32 array.
               nt matches d_obs_baseline.shape[1].
    """
    n_ts = len(src_indices)
    nt = d_obs_baseline.shape[1]
    wavelets = np.zeros((n_ts, nt), dtype=np.float64)

    for i, si in enumerate(src_indices):
        stack = np.zeros(nt, dtype=np.float64)
        n_stacked = 0
        for ri in rec_indices:
            pair_idx = int(si) * n_receivers + int(ri)
            if pair_idx >= d_obs_baseline.shape[0]:
                continue
            tr = d_obs_baseline[pair_idx, :].astype(np.float64)
            pick = int(baseline_picks[pair_idx])
            i0 = max(pick - gate_pre_samples, 0)
            i1 = min(pick + gate_post_samples, nt)
            windowed = np.zeros(nt, dtype=np.float64)
            windowed[i0:i1] = tr[i0:i1]
            rms = float(np.sqrt(np.mean(windowed[i0:i1] ** 2))) if i1 > i0 else 0.0
            if rms > 0:
                stack += windowed / rms  # normalise each trace before stacking
                n_stacked += 1

        if n_stacked > 0:
            peak = float(np.max(np.abs(stack)))
            wavelets[i] = stack / max(peak, 1e-30)
        else:
            # Fallback: Ricker wavelet at 5 kHz (generic CUSSP frequency)
            wavelets[i] = _ricker_wavelet(5000.0, 1.0 / 48000.0, nt)

    return wavelets.astype(np.float32)


# ---------------------------------------------------------------------------
# FATT initial velocity model
# ---------------------------------------------------------------------------

def build_initial_vp(
    grid: FWIGrid,
    src_pos_grid: np.ndarray,     # (n_src, 2) ix,iz grid indices
    rec_pos_grid: np.ndarray,     # (n_rec, 2) ix,iz grid indices
    baseline_picks_samples: np.ndarray,  # (n_src * n_rec,) subset for DM*→TS pairs
    src_indices: np.ndarray,      # (n_src,) global source indices
    rec_indices: np.ndarray,      # (n_rec,) global receiver indices
    n_receivers: int,
    sample_rate_hz: float,
    vp_background: float = 3000.0,
) -> np.ndarray:
    """Build initial 2D v_p model via straight-ray FATT back-projection.

    Uses only the DM*→TS pair picks supplied in baseline_picks_samples.
    Rays are straight lines from source to receiver grid positions.
    The back-projected slowness perturbation is added to a uniform
    background model.

    If the picks are too sparse or noisy to constrain the model, returns
    the uniform background model.

    Returns
    -------
    vp : (nz, nx) float64 array in m/s
    """
    nz, nx = grid.nz, grid.nx
    vp = np.full((nz, nx), vp_background, dtype=np.float64)

    # Accumulate hit count and slowness residual per cell
    hit_count = np.zeros((nz, nx), dtype=np.float64)
    slowness_update = np.zeros((nz, nx), dtype=np.float64)

    n_src = len(src_indices)
    n_rec_ts = len(rec_indices)
    valid_count = 0

    for i_src, si in enumerate(src_indices):
        sx, sz = int(src_pos_grid[i_src, 0]), int(src_pos_grid[i_src, 1])
        src_world_x = grid.x[sx]
        src_world_z = grid.z[sz]

        for i_rec, ri in enumerate(rec_indices):
            pair_flat = int(si) * n_receivers + int(ri)
            if pair_flat >= len(baseline_picks_samples):
                continue
            pick_s = int(baseline_picks_samples[pair_flat]) / sample_rate_hz

            rx, rz = int(rec_pos_grid[i_rec, 0]), int(rec_pos_grid[i_rec, 1])
            rec_world_x = grid.x[rx]
            rec_world_z = grid.z[rz]

            dist_m = math.sqrt(
                (rec_world_x - src_world_x) ** 2 + (rec_world_z - src_world_z) ** 2
            )
            if dist_m < 0.1:
                continue

            # Predicted traveltime from background model
            t_pred = dist_m / vp_background
            dt_res = pick_s - t_pred  # positive = slower than background

            if abs(dt_res) > 0.5 * t_pred:
                # Residual >50% of predicted — skip as outlier
                continue

            # Straight-ray back-projection using Bresenham-like line rasterisation
            n_steps = max(int(math.ceil(
                math.sqrt((rx - sx) ** 2 + (rz - sz) ** 2)
            )) + 1, 2)
            xs_ray = np.round(np.linspace(sx, rx, n_steps)).astype(int)
            zs_ray = np.round(np.linspace(sz, rz, n_steps)).astype(int)
            xs_ray = np.clip(xs_ray, 0, nx - 1)
            zs_ray = np.clip(zs_ray, 0, nz - 1)

            seg_len = dist_m / n_steps  # approximate segment length per cell
            for xi, zi in zip(xs_ray, zs_ray):
                # Slowness residual spread uniformly along the ray
                slowness_update[zi, xi] += (dt_res / dist_m) * seg_len
                hit_count[zi, xi] += seg_len

            valid_count += 1

    if valid_count < 3:
        LOG.warning(
            "FATT: fewer than 3 valid pick residuals (%d); returning uniform background v_p=%.0f m/s.",
            valid_count, vp_background,
        )
        return vp

    # Convert accumulated slowness residual to velocity perturbation
    mask = hit_count > 0
    mean_slowness = slowness_update[mask] / hit_count[mask]
    # δv_p ≈ -v_background² * δs, where δs = δt/dist (slowness residual)
    vp[mask] = vp_background / (1.0 + vp_background * mean_slowness / 1.0)
    vp = np.clip(vp, 500.0, 10000.0)

    LOG.info(
        "FATT: initialised v_p from %d valid pair picks; range %.0f–%.0f m/s.",
        valid_count, float(vp.min()), float(vp.max()),
    )
    return vp


# ---------------------------------------------------------------------------
# Correlative misfit: 1 - NCC for a single pair-epoch
# ---------------------------------------------------------------------------

def _ncc_and_lag(
    d_m: np.ndarray,    # (nt,) synthetic
    d_obs: np.ndarray,  # (nt,) observed
) -> Tuple[float, float]:
    """Normalised cross-correlation peak and its lag (samples).

    Returns (peak_ncc, lag_samples) where lag_samples > 0 means d_obs
    arrives LATER than d_m (positive dt → slowdown).
    """
    from scipy.signal import correlate as _correlate
    b = d_m.astype(np.float64)
    e = d_obs.astype(np.float64)
    b_norm = np.linalg.norm(b)
    e_norm = np.linalg.norm(e)
    if b_norm < 1e-30 or e_norm < 1e-30:
        return 0.0, 0.0
    cc = _correlate(e / e_norm, b / b_norm, mode="full")
    center = len(cc) // 2
    peak_idx = int(np.argmax(cc))
    peak_ncc = float(cc[peak_idx])
    lag_int = peak_idx - center
    # Parabolic sub-sample refinement
    if 0 < peak_idx < len(cc) - 1:
        y0, y1, y2 = cc[peak_idx - 1], cc[peak_idx], cc[peak_idx + 1]
        denom = 2.0 * y1 - y0 - y2
        sub = float(np.clip((y2 - y0) / (2.0 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        sub = 0.0
    return peak_ncc, float(lag_int) + sub


def _correlative_misfit_1d(d_m: np.ndarray, d_obs: np.ndarray) -> float:
    """Return 1 - NCC(d_m, d_obs).  Range [0, 2]; 0 = perfect phase match."""
    ncc, _ = _ncc_and_lag(d_m, d_obs)
    return 1.0 - ncc


# ---------------------------------------------------------------------------
# 1D line search over dt — the core of the Phase 1 approach
# ---------------------------------------------------------------------------

def _line_search_dt(
    d_m: np.ndarray,          # (nt,) synthetic at t=0 (no shift applied)
    d_obs: np.ndarray,        # (nt,) observed waveform window
    dt_grid: float,           # model time step (s)
    sample_rate_hz: float,    # observed data sample rate (s)
    dt_search_max_s: float,   # ±search range in seconds
    dt_center_s: float = 0.0, # center of the search window in seconds
    min_ncc: float = 0.2,
) -> Tuple[float, float, bool]:
    """Find dt* = argmin_dt (1 - NCC(d_m(t + dt), d_obs)).

    The synthetic d_m is time-shifted by candidate dt values via
    integer-sample shift (sub-sample precision added via parabolic
    refinement around the best integer shift).

    Returns
    -------
    dt_us      : best-fit time shift in microseconds
                 (positive = d_obs arrives later than d_m → slowdown)
    peak_ncc   : NCC at the best-fit shift
    rejected   : True if peak_ncc < min_ncc or arrays are degenerate
    """
    nt = len(d_m)
    dt_data = 1.0 / sample_rate_hz
    n_obs = len(d_obs)

    if nt < 1 or n_obs < 1:
        return 0.0, 0.0, True

    # Work on the data sample grid so lag samples and the returned dt are
    # always expressed in observed-data units.
    t_syn = np.arange(nt, dtype=np.float64) * float(dt_grid)
    t_dat = np.arange(n_obs, dtype=np.float64) * dt_data

    # Resample the synthetic onto the data sample grid, then zero-pad outside
    # the model support. This keeps the line search deterministic even when
    # the solver grid step differs from the observed sample interval.
    if abs(dt_grid - dt_data) > dt_grid * 0.05:
        LOG.debug(
            "FWI line search: synthetic dt=%.2e s, data dt=%.2e s — resampling synthetic.",
            dt_grid, dt_data,
        )
        d_m_work = np.interp(t_dat, t_syn, d_m, left=0.0, right=0.0).astype(d_m.dtype, copy=False)
    else:
        # Same sample step: truncate or zero-pad to match d_obs length.
        d_m_work = np.zeros(n_obs, dtype=d_m.dtype)
        copy_len = min(nt, n_obs)
        d_m_work[:copy_len] = d_m[:copy_len]

    # Convert search range to observed-data samples and center the search on
    # the previous band's estimate rather than shifting the waveform itself.
    center_shift_samples = int(round(float(dt_center_s) / dt_data)) if abs(dt_center_s) > 0.0 else 0
    max_shift_samples = max(int(round(dt_search_max_s / dt_data)), 1)

    # Build shifted versions by integer roll; evaluate NCC at each shift
    shifts = center_shift_samples + np.arange(-max_shift_samples, max_shift_samples + 1)
    ncc_vals = np.zeros(len(shifts), dtype=np.float64)

    d_m_norm = np.linalg.norm(d_m_work)
    d_obs_norm = np.linalg.norm(d_obs)
    if d_m_norm < 1e-30 or d_obs_norm < 1e-30:
        return 0.0, 0.0, True

    d_m_n = d_m_work.astype(np.float64) / d_m_norm
    d_obs_n = d_obs.astype(np.float64) / d_obs_norm

    for k, s in enumerate(shifts):
        if s == 0:
            d_shifted = d_m_n
        else:
            d_shifted = np.roll(d_m_n, s)
            # Zero out wrap-around region
            if s > 0:
                d_shifted[:s] = 0.0
            else:
                d_shifted[s:] = 0.0
        ncc_vals[k] = float(np.dot(d_shifted, d_obs_n))

    best_k = int(np.argmax(ncc_vals))
    peak_ncc = float(ncc_vals[best_k])
    best_shift = shifts[best_k]

    # Parabolic sub-sample refinement
    if 0 < best_k < len(ncc_vals) - 1:
        y0, y1, y2 = ncc_vals[best_k - 1], ncc_vals[best_k], ncc_vals[best_k + 1]
        denom = 2.0 * y1 - y0 - y2
        sub = float(np.clip((y2 - y0) / (2.0 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        sub = 0.0

    # dt* in seconds (shift in observed-data samples × observed-data dt)
    dt_star_s = (float(best_shift) + sub) * dt_data
    dt_us = dt_star_s * 1e6

    rejected = peak_ncc < min_ncc
    return dt_us, peak_ncc, rejected


# ---------------------------------------------------------------------------
# Multiscale FWI dt loop
# ---------------------------------------------------------------------------

def multiscale_fwi_dt_loop(
    bl_win: np.ndarray,              # (n_samples,) baseline waveform window (preprocessed)
    ep_win: np.ndarray,              # (n_samples,) epoch waveform window
    vp: np.ndarray,                  # (nz, nx) velocity model
    grid: FWIGrid,
    source_wavelet: np.ndarray,      # (nt,) estimated source wavelet for this source
    src_ix: int,
    src_iz: int,
    rec_ix: int,
    rec_iz: int,
    solver: SolverBackend,
    freq_bands: List[Tuple[float, float]],  # [(f_low, f_high), ...] Hz, coarse → fine
    sample_rate_hz: float,
    dt_search_max_s: float = 0.002,
    min_ncc: float = 0.2,
    filter_order: int = 4,
) -> Tuple[float, float]:
    """Run FWI dt line search across frequency bands from coarse to fine.

    Each band narrows the search window around the previous band's result.
    The synthetic is computed once per band (the forward model is band-
    dependent because the source wavelet is bandpassed before injection).

    Returns
    -------
    dt_us    : best-fit travel-time shift in microseconds
    best_ncc : NCC at the best-fit shift for the finest band
    """
    dt_current_max_s = float(dt_search_max_s)
    dt_best_s = 0.0
    dt_center_s = 0.0
    ncc_best = 0.0

    for band_idx, (f_low, f_high) in enumerate(freq_bands):
        # Bandpass both observed windows and the source wavelet
        bl_band = _bandpass(bl_win.astype(np.float32), f_low, f_high,
                            sample_rate_hz, order=filter_order)
        ep_band = _bandpass(ep_win.astype(np.float32), f_low, f_high,
                            sample_rate_hz, order=filter_order)
        wav_band = _bandpass(source_wavelet.astype(np.float32), f_low, f_high,
                             1.0 / grid.dt, order=filter_order)

        # Forward model with bandpassed wavelet
        try:
            d_m_all = solver.forward(
                vp=vp,
                source_wavelet=wav_band.astype(np.float64),
                src_ix=src_ix,
                src_iz=src_iz,
                rec_ix=np.array([rec_ix], dtype=int),
                rec_iz=np.array([rec_iz], dtype=int),
                grid=grid,
            )  # (1, nt)
            d_m = d_m_all[0].astype(np.float32)
        except Exception as exc:
            LOG.warning("FWI forward model failed for band %d: %s", band_idx, exc)
            continue

        # Bandpass the synthetic at the same band for fair comparison
        d_m_band = _bandpass(d_m, f_low, f_high, 1.0 / grid.dt, order=filter_order)

        # 1D line search
        dt_us, ncc, rejected = _line_search_dt(
            d_m=d_m_band.astype(np.float64),
            d_obs=ep_band.astype(np.float64),
            dt_grid=grid.dt,
            sample_rate_hz=sample_rate_hz,
            dt_search_max_s=dt_current_max_s,
            dt_center_s=dt_center_s,
            min_ncc=0.0,  # don't reject mid-loop; check at end
        )

        if ncc > ncc_best:
            dt_best_s = dt_us * 1e-6
            ncc_best = ncc

        if not rejected:
            dt_center_s = dt_best_s

        # Narrow search window for next band: ±half-period at this band's f_low
        # (the half-period defines the cycle-skip threshold)
        if f_low > 0:
            half_period = 0.5 / f_low
        else:
            half_period = dt_current_max_s
        dt_current_max_s = min(dt_current_max_s, half_period)

        LOG.debug(
            "FWI dt band %d [%.0f–%.0f Hz]: dt=%.1f µs, NCC=%.3f, next_max=%.3f ms",
            band_idx, f_low, f_high, dt_best_s * 1e6, ncc_best, dt_current_max_s * 1e3,
        )

    rejected = ncc_best < min_ncc
    dt_best_us = dt_best_s * 1e6
    return dt_best_us, ncc_best


# ---------------------------------------------------------------------------
# Top-level per-pair FWI dt estimator (drop-in for _xcorr_dt_samples)
# ---------------------------------------------------------------------------

def fwi_estimate_dt(
    bl_win: np.ndarray,
    ep_win: np.ndarray,
    vp: np.ndarray,
    grid: FWIGrid,
    source_wavelet: np.ndarray,
    src_ix: int,
    src_iz: int,
    rec_ix: int,
    rec_iz: int,
    solver: SolverBackend,
    freq_bands: List[Tuple[float, float]],
    sample_rate_hz: float,
    dt_search_max_s: float = 0.002,
    min_ncc: float = 0.2,
) -> Tuple[float, float, bool]:
    """FWI-derived travel-time shift for a single source-receiver pair and epoch.

    Drop-in replacement for _xcorr_dt_samples() in compute_metrics().
    Returns (dt_us, peak_ncc, rejected) with the same sign convention:
      dt_us > 0 → epoch arrives LATER than baseline (slowdown, v_p decreased).

    Algorithm
    ---------
    1. Run multiscale_fwi_dt_loop() coarse-to-fine across freq_bands.
       Each band bandpasses the observed windows and the source wavelet,
       runs the forward model, then does a 1D line search for the NCC
       maximum over ±dt_search_max_s (narrowing each band).
    2. Return the overall best (dt, NCC) from all bands.
    3. Mark as rejected if peak_ncc < min_ncc.

    Parameters
    ----------
    bl_win, ep_win     : preprocessed + tapered waveform windows (same as fed
                         to _xcorr_dt_samples in compute_metrics)
    vp                 : 2D velocity model (nz, nx) in m/s from build_initial_vp()
    grid               : FWIGrid for the DM-borehole → TS-string cross-section
    source_wavelet     : (nt,) estimated wavelet for this source (from
                         estimate_source_wavelet())
    src_ix, src_iz     : grid indices of the source (DM borehole position)
    rec_ix, rec_iz     : grid indices of the receiver (TS hydrophone position)
    solver             : SolverBackend instance (AnalyticSolver or FD2DSolver)
    freq_bands         : list of (f_low, f_high) tuples, coarse to fine
    sample_rate_hz     : data sample rate (Hz) for bandpass filtering
    dt_search_max_s    : initial ±search range for the coarsest band (s)
    min_ncc            : minimum NCC to accept the result

    Returns
    -------
    dt_us    : float — travel-time shift in microseconds
    peak_ncc : float — normalised cross-correlation at the best shift
    rejected : bool  — True if peak_ncc < min_ncc or arrays are degenerate
    """
    if bl_win.size < 4 or ep_win.size < 4:
        return 0.0, 0.0, True

    try:
        dt_us, ncc = multiscale_fwi_dt_loop(
            bl_win=bl_win,
            ep_win=ep_win,
            vp=vp,
            grid=grid,
            source_wavelet=source_wavelet,
            src_ix=src_ix,
            src_iz=src_iz,
            rec_ix=rec_ix,
            rec_iz=rec_iz,
            solver=solver,
            freq_bands=freq_bands,
            sample_rate_hz=sample_rate_hz,
            dt_search_max_s=dt_search_max_s,
            min_ncc=min_ncc,
        )
    except Exception as exc:
        LOG.warning("fwi_estimate_dt: exception during forward/line-search: %s", exc)
        return 0.0, 0.0, True

    rejected = ncc < min_ncc
    return float(dt_us), float(ncc), rejected


# ---------------------------------------------------------------------------
# FWI context: pre-computation shared across all DM*→TS pairs in one run
# ---------------------------------------------------------------------------

@dataclass
class FWIContext:
    """Pre-computed FWI objects shared across all DM*→TS pair-epoch computations.

    Built once per processing run (in run_once() before the pair loop) and
    passed into compute_metrics() so the solver, grid, velocity model, and
    source wavelets are not reconstructed per pair.

    Attributes
    ----------
    grid            : FWIGrid for the DM-borehole → TS cross-section
    vp              : (nz, nx) float64 baseline velocity model from FATT
    solver          : SolverBackend instance
    source_wavelets : (n_dm_sources, nt) per-source estimated wavelets
    src_pos_grid    : (n_dm_sources, 2) ix,iz grid indices for DM sources
    rec_pos_grid    : (n_ts_receivers, 2) ix,iz grid indices for TS receivers
    src_global_idx  : (n_dm_sources,) global 0-based source indices
    rec_global_idx  : (n_ts_receivers,) global 0-based receiver indices
    freq_bands      : list of (f_low, f_high) tuples used in multiscale loop
    dt_search_max_s : initial ±dt search range (s)
    min_ncc         : minimum NCC acceptance threshold
    n_receivers     : total number of receivers in the full geometry (for pair_idx computation)
    axis_unit       : (2,) horizontal unit vector defining the 2D plane
    """
    grid: FWIGrid
    vp: np.ndarray
    solver: "SolverBackend"
    source_wavelets: np.ndarray          # (n_dm_sources, nt)
    src_pos_grid: np.ndarray             # (n_dm_sources, 2)  ix,iz
    rec_pos_grid: np.ndarray             # (n_ts_receivers, 2) ix,iz
    src_global_idx: np.ndarray           # (n_dm_sources,)
    rec_global_idx: np.ndarray           # (n_ts_receivers,)
    freq_bands: List[Tuple[float, float]]
    dt_search_max_s: float
    min_ncc: float
    n_receivers: int
    axis_unit: np.ndarray                # (2,) for projecting new positions if needed

    def get_pair_grid_pos(
        self, global_src_idx: int, global_rec_idx: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Return (src_ix, src_iz), (rec_ix, rec_iz) for a global pair index.

        Returns (None, None) if the source or receiver is not in the DM*→TS set.
        """
        src_local = np.where(self.src_global_idx == global_src_idx)[0]
        rec_local = np.where(self.rec_global_idx == global_rec_idx)[0]
        if src_local.size == 0 or rec_local.size == 0:
            return None, None
        si = int(src_local[0])
        ri = int(rec_local[0])
        src_pos = (int(self.src_pos_grid[si, 0]), int(self.src_pos_grid[si, 1]))
        rec_pos = (int(self.rec_pos_grid[ri, 0]), int(self.rec_pos_grid[ri, 1]))
        return src_pos, rec_pos

    def get_source_wavelet(self, global_src_idx: int) -> np.ndarray:
        """Return the estimated wavelet for a given global source index."""
        src_local = np.where(self.src_global_idx == global_src_idx)[0]
        if src_local.size == 0:
            # Fallback to Ricker at 5 kHz
            return _ricker_wavelet(5000.0, self.grid.dt, self.grid.nt)
        return self.source_wavelets[int(src_local[0])]


def build_fwi_context(
    tg_n_sources: int,
    tg_n_receivers: int,
    tg_sample_rate_hz: float,
    tg_sample_count: int,
    d_obs_baseline: np.ndarray,      # (n_pairs, sample_count) averaged baseline waveforms
    baseline_picks: np.ndarray,      # (n_pairs,) from _baseline_picks()
    source_boreholes: List[str],     # length n_sources; e.g. ["AML","AML",...,"DML","DML",...]
    sources_csv: Path,
    receivers_csv: Path,
    solver_name: str = "fd2d",
    grid_dx_m: float = 0.5,
    grid_dz_m: float = 0.5,
    grid_padding_m: float = 20.0,
    vp_background_mps: float = 3000.0,
    freq_bands: Optional[List[Tuple[float, float]]] = None,
    dt_search_max_ms: float = 2.0,
    min_ncc: float = 0.2,
    gate_pre_ms: Optional[float] = None,
    gate_post_ms: Optional[float] = None,
    cpml_thickness: int = 20,
) -> "FWIContext":
    """Build an FWIContext from the current CASSMTempGather state.

    This is called once per processing run (before the pair loop) to
    pre-compute the grid, velocity model, solver, and source wavelets.

    Parameters
    ----------
    tg_n_sources, tg_n_receivers, tg_sample_rate_hz, tg_sample_count
        : geometry from CASSMTempGather
    d_obs_baseline   : (n_pairs, sample_count) averaged baseline epoch data
    baseline_picks   : (n_pairs,) sample indices for the P-arrival picks
    source_boreholes : list of borehole labels per source (same as MetricConfig.source_boreholes)
    sources_csv, receivers_csv : coordinate files (HMC frame)
    solver_name      : "analytic", "fd2d", or "devito"
    grid_dx_m etc.   : FWI grid parameters from config fwi_dt: section
    freq_bands       : list of (f_low, f_high) Hz tuples; defaults to standard CUSSP bands
    dt_search_max_ms : initial search range in milliseconds
    min_ncc          : NCC acceptance threshold
    gate_pre_ms, gate_post_ms : P-wave gate for wavelet estimation (ms)
    cpml_thickness   : CPML layers for FD2DSolver

    Returns
    -------
    FWIContext ready to use in compute_metrics()
    """
    import pandas as pd

    def _require_csv_file(path: Path, label: str) -> Path:
        if str(path).strip() in ("", "."):
            raise ValueError(
                f"build_fwi_context: {label} is empty; set fwi_dt.{label} to a CSV file path."
            )
        if path.exists() and path.is_dir():
            raise IsADirectoryError(
                f"build_fwi_context: {label} points to a directory, not a CSV file: {path}"
            )
        if not path.exists():
            raise FileNotFoundError(
                f"build_fwi_context: {label} does not exist: {path}"
            )
        return path

    if freq_bands is None:
        freq_bands = [(250.0, 2000.0), (500.0, 8000.0), (1000.0, 20000.0)]

    if len(source_boreholes) < tg_n_sources:
        raise ValueError(
            "build_fwi_context: source_boreholes is shorter than tg_n_sources "
            f"({len(source_boreholes)} < {tg_n_sources}); FWI needs one borehole label per source."
        )

    # --- Identify DM* source indices and TS receiver indices ---
    dm_src_indices = np.array(
        [i for i, bh in enumerate(source_boreholes[:tg_n_sources])
         if bh.upper().startswith("DM")],
        dtype=np.int32,
    )
    ts_rec_indices = np.arange(48, tg_n_receivers, dtype=np.int32)  # ch 49-72 (0-based 48-71)

    if dm_src_indices.size == 0:
        raise ValueError(
            "build_fwi_context: no DM* sources found in source_boreholes list. "
            "FWI dt is only applicable to DM*→TS hydrophone pairs."
        )
    if ts_rec_indices.size == 0:
        raise ValueError(
            "build_fwi_context: no TS hydrophone receivers found (expected rec idx 48+)."
        )

    # --- Load source/receiver coordinates ---
    sources_csv = _require_csv_file(Path(sources_csv), "sources_csv")
    receivers_csv = _require_csv_file(Path(receivers_csv), "receivers_csv")
    src_df = pd.read_csv(sources_csv)
    rec_df = pd.read_csv(receivers_csv)

    def _require_columns(df: "pd.DataFrame", required: Sequence[str], label: str) -> None:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"build_fwi_context: {label} CSV is missing required columns: {', '.join(missing)}"
            )

    def _load_dm_source_xyz() -> Tuple[np.ndarray, np.ndarray]:
        _require_columns(src_df, ("source_id", "borehole", "source_index", "x", "y", "z"), "sources")

        src_work = src_df.copy()
        src_work["borehole"] = src_work["borehole"].astype(str).str.upper().str.strip()
        src_work["source_id"] = src_work["source_id"].astype(str).str.strip()
        src_work["source_index"] = pd.to_numeric(src_work["source_index"], errors="coerce")
        for col in ("x", "y", "z"):
            src_work[col] = pd.to_numeric(src_work[col], errors="coerce")
        src_work = src_work.dropna(subset=["source_id", "borehole", "source_index", "x", "y", "z"])
        src_work["source_index"] = src_work["source_index"].astype(int)

        source_rows_by_borehole: Dict[str, List[Tuple[str, int, np.ndarray]]] = {}
        for borehole, group in src_work.groupby("borehole", sort=False):
            ordered = group.sort_values(["source_index", "source_id"], kind="stable")
            source_rows_by_borehole[borehole] = [
                (
                    str(row.source_id),
                    int(row.source_index),
                    np.array([float(row.x), float(row.y), float(row.z)], dtype=np.float64),
                )
                for row in ordered.itertuples(index=False)
            ]

        dm_xyz: List[np.ndarray] = []
        dm_global_indices: List[int] = []
        borehole_counts: Dict[str, int] = {}
        missing_dm: List[str] = []

        for global_src_idx, borehole in enumerate(source_boreholes[:tg_n_sources]):
            borehole_norm = str(borehole).upper().strip()
            if not borehole_norm.startswith("DM"):
                continue

            ordinal = borehole_counts.get(borehole_norm, 0)
            borehole_counts[borehole_norm] = ordinal + 1

            entries = source_rows_by_borehole.get(borehole_norm, [])
            if ordinal >= len(entries):
                missing_dm.append(f"{borehole_norm} source_index={ordinal + 1}")
                continue

            _source_id, _source_index, xyz = entries[ordinal]
            dm_global_indices.append(global_src_idx)
            dm_xyz.append(xyz)

        if not dm_global_indices:
            raise ValueError(
                "build_fwi_context: no DM* source rows found in sources_csv for the configured source_boreholes."
            )
        if missing_dm:
            raise ValueError(
                "build_fwi_context: missing DM source rows in sources_csv: "
                + ", ".join(missing_dm)
            )

        return np.asarray(dm_xyz, dtype=np.float64), np.asarray(dm_global_indices, dtype=np.int32)

    def _load_ts_receiver_xyz() -> np.ndarray:
        _require_columns(rec_df, ("receiver_id", "x", "y", "z"), "receivers")

        rec_work = rec_df.copy()
        rec_work["receiver_id"] = rec_work["receiver_id"].astype(str).str.upper().str.strip()
        for col in ("x", "y", "z"):
            rec_work[col] = pd.to_numeric(rec_work[col], errors="coerce")
        rec_work = rec_work.dropna(subset=["receiver_id", "x", "y", "z"])

        rec_by_id: Dict[str, np.ndarray] = {}
        for row in rec_work.itertuples(index=False):
            rec_by_id[str(row.receiver_id)] = np.array(
                [float(row.x), float(row.y), float(row.z)], dtype=np.float64
            )

        ts_xyz: List[np.ndarray] = []
        missing_ts: List[str] = []
        for rec_idx in range(48, tg_n_receivers):
            rec_id = f"TS{rec_idx - 47:02d}"
            xyz = rec_by_id.get(rec_id)
            if xyz is None:
                missing_ts.append(rec_id)
                continue
            ts_xyz.append(xyz)

        if not ts_xyz:
            raise ValueError(
                "build_fwi_context: no TS receiver rows found in receivers_csv for the configured receiver count."
            )
        if missing_ts:
            raise ValueError(
                "build_fwi_context: missing TS receiver rows in receivers_csv: "
                + ", ".join(missing_ts)
            )

        return np.asarray(ts_xyz, dtype=np.float64)

    dm_src_xyz, dm_src_global_idx = _load_dm_source_xyz()
    ts_rec_xyz = _load_ts_receiver_xyz()

    # --- Compute inter-borehole axis unit vector ---
    src_mean_horiz = dm_src_xyz[:, :2].mean(axis=0)
    rec_mean_horiz = ts_rec_xyz[:, :2].mean(axis=0)
    axis = rec_mean_horiz - src_mean_horiz
    axis_len = np.linalg.norm(axis)
    axis_unit = axis / axis_len if axis_len > 1e-3 else np.array([1.0, 0.0])

    # --- Build FWI grid ---
    record_time_s = tg_sample_count / tg_sample_rate_hz
    grid = FWIGrid.from_source_receiver_positions(
        src_xyz=dm_src_xyz,
        rec_xyz=ts_rec_xyz,
        dx=grid_dx_m,
        dz=grid_dz_m,
        dt=None,  # auto CFL
        vp_max_estimate=vp_background_mps * 1.5,
        record_time_s=record_time_s,
        padding_m=grid_padding_m,
    )
    LOG.info(
        "FWI grid: %d×%d (nx×nz), dx=%.2fm, dz=%.2fm, dt=%.2eµs, nt=%d",
        grid.nx, grid.nz, grid.dx, grid.dz, grid.dt * 1e6, grid.nt,
    )

    # --- Project source/receiver positions onto 2D grid ---
    src_pos_grid = grid.project_positions(dm_src_xyz, axis_unit)   # (n_dm, 2) ix,iz
    rec_pos_grid = grid.project_positions(ts_rec_xyz, axis_unit)   # (n_ts, 2) ix,iz

    # --- Select solver ---
    if solver_name == "analytic":
        solver: SolverBackend = AnalyticSolver()
        LOG.info("FWI using AnalyticSolver (straight-ray + Ricker, for testing only).")
    elif solver_name == "devito":
        try:
            from cussp_cassm_fwi_devito import DevitoSolver  # type: ignore
            solver = DevitoSolver()
            LOG.info("FWI using DevitoSolver.")
        except ImportError:
            LOG.warning("Devito not available; falling back to FD2DSolver.")
            solver = FD2DSolver(cpml_thickness=cpml_thickness)
    else:
        solver = FD2DSolver(cpml_thickness=cpml_thickness)
        LOG.info("FWI using FD2DSolver (O(2,2) explicit acoustic FD).")

    # --- FATT initial velocity model ---
    vp = build_initial_vp(
        grid=grid,
        src_pos_grid=src_pos_grid,
        rec_pos_grid=rec_pos_grid,
        baseline_picks_samples=baseline_picks,
        src_indices=dm_src_indices,
        rec_indices=ts_rec_indices,
        n_receivers=tg_n_receivers,
        sample_rate_hz=tg_sample_rate_hz,
        vp_background=vp_background_mps,
    )

    # --- Estimate source wavelets ---
    gate_pre_s = gate_pre_ms / 1000.0 if gate_pre_ms is not None else 0.001
    gate_post_s = gate_post_ms / 1000.0 if gate_post_ms is not None else 0.002
    gate_pre_samples = max(int(gate_pre_s * tg_sample_rate_hz), 1)
    gate_post_samples = max(int(gate_post_s * tg_sample_rate_hz), 1)

    # Use only valid DM* source indices that exist in both the geometry and data
    valid_dm_src = dm_src_indices[dm_src_indices < tg_n_sources]
    source_wavelets = estimate_source_wavelet(
        d_obs_baseline=d_obs_baseline,
        baseline_picks=baseline_picks,
        gate_pre_samples=gate_pre_samples,
        gate_post_samples=gate_post_samples,
        src_indices=valid_dm_src,
        rec_indices=ts_rec_indices[ts_rec_indices < tg_n_receivers],
        n_receivers=tg_n_receivers,
    )

    # Resample each wavelet from the data time axis to the grid time axis.
    n_wav, wav_len = source_wavelets.shape
    if wav_len != grid.nt:
        data_t = np.arange(wav_len, dtype=np.float64) / tg_sample_rate_hz
        grid_t = np.arange(grid.nt, dtype=np.float64) * grid.dt
        resampled = np.zeros((n_wav, grid.nt), dtype=np.float64)
        for i in range(n_wav):
            resampled[i] = np.interp(grid_t, data_t, source_wavelets[i], left=0.0, right=0.0)
        source_wavelets = resampled.astype(np.float32)

    ctx = FWIContext(
        grid=grid,
        vp=vp,
        solver=solver,
        source_wavelets=source_wavelets,
        src_pos_grid=src_pos_grid,
        rec_pos_grid=rec_pos_grid,
        src_global_idx=dm_src_global_idx,
        rec_global_idx=ts_rec_indices,
        freq_bands=freq_bands,
        dt_search_max_s=dt_search_max_ms / 1000.0,
        min_ncc=min_ncc,
        n_receivers=tg_n_receivers,
        axis_unit=axis_unit,
    )
    LOG.info(
        "FWIContext built: %d DM* sources, %d TS receivers, v_p range %.0f–%.0f m/s.",
        valid_dm_src.size,
        ctx.rec_global_idx.size,
        float(vp.min()),
        float(vp.max()),
    )
    return ctx


# ---------------------------------------------------------------------------
# Phase 2 stubs — full CDD-TLFWI δv_p spatial inversion (not yet called)
# ---------------------------------------------------------------------------

def build_observed_baseline(
    data: np.ndarray,        # (n_epochs, n_pairs, n_samples)
    baseline_n_epochs: int,
) -> np.ndarray:
    """Average first baseline_n_epochs epochs → d_obs_baseline (n_pairs, n_samples).

    Phase 2 stub — not called from the current Phase 1 pipeline.
    """
    n_base = min(baseline_n_epochs, data.shape[0])
    if n_base < 1:
        return np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
    return np.mean(data[:n_base], axis=0).astype(np.float32)


def build_pseudo_monitoring_data(
    d_m_baseline: np.ndarray,    # (n_pairs, nt) modelled baseline
    d_obs_baseline: np.ndarray,  # (n_pairs, nt) observed baseline
    d_obs_epoch: np.ndarray,     # (n_pairs, nt) observed monitor epoch
    valid_pairs_mask: np.ndarray,
    n_sources: int,
    n_receivers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct pseudo monitoring data per Eq. 2 of the paper.

        δd    = d_obs_epoch - d_obs_baseline   (waveform difference)
        w(xs) = max|d_obs_baseline[src,:]| / max|d_m_baseline[src,:]|  per-shot scale
        d_pm  = w * d_m_baseline + δd

    Phase 2 stub — not called from the current Phase 1 pipeline.

    Returns (d_pm, w_per_source) shapes (n_pairs, nt) and (n_sources,).
    """
    n_pairs, nt = d_obs_epoch.shape
    delta_d = d_obs_epoch - d_obs_baseline
    w_per_source = np.ones(n_sources, dtype=np.float64)
    for si in range(n_sources):
        pair_start = si * n_receivers
        pair_end = min(pair_start + n_receivers, n_pairs)
        obs_rows = d_obs_baseline[pair_start:pair_end]
        mod_rows = d_m_baseline[pair_start:pair_end]
        obs_max = float(np.max(np.abs(obs_rows))) if obs_rows.size else 0.0
        mod_max = float(np.max(np.abs(mod_rows))) if mod_rows.size else 0.0
        if mod_max > 1e-30:
            w_per_source[si] = obs_max / mod_max

    d_pm = np.zeros_like(d_obs_epoch, dtype=np.float32)
    for si in range(n_sources):
        pair_start = si * n_receivers
        pair_end = min(pair_start + n_receivers, n_pairs)
        d_pm[pair_start:pair_end] = (
            w_per_source[si] * d_m_baseline[pair_start:pair_end]
            + delta_d[pair_start:pair_end]
        )
    return d_pm, w_per_source.astype(np.float32)


def correlative_misfit(
    d_m: np.ndarray,              # (n_pairs, nt)
    d_obs: np.ndarray,            # (n_pairs, nt)
    valid_pairs_mask: np.ndarray, # (n_pairs,) bool
    min_ncc: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """Compute the correlative misfit f = Σ_pairs (1 - NCC(d_m, d_obs)).

    Phase 2 stub — not called from the current Phase 1 pipeline.

    Returns
    -------
    total_misfit : float — sum of per-pair (1 - NCC)
    per_pair_ncc : (n_pairs,) float32 array of NCC values (NaN for invalid pairs)
    """
    n_pairs = d_m.shape[0]
    per_pair_ncc = np.full(n_pairs, np.nan, dtype=np.float32)
    total_misfit = 0.0
    n_valid = 0

    for p in range(n_pairs):
        if not valid_pairs_mask[p]:
            continue
        ncc, _ = _ncc_and_lag(d_m[p], d_obs[p])
        per_pair_ncc[p] = float(ncc)
        if ncc >= min_ncc:
            total_misfit += 1.0 - float(ncc)
            n_valid += 1

    return total_misfit, per_pair_ncc
