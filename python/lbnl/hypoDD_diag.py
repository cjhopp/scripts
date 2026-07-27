#!/usr/bin/env python3
"""
hypodd_plot.py - Interactive HypoDD relocation plotter + diagnostics
Assemble parts 1-6 in order into a single file.

Usage: python hypodd_plot.py [--reloc FILE] [--loc FILE] [--sta FILE] [--line FILE] [--clusters]
Requirements: numpy matplotlib (optional: cartopy for enhanced borders)
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # change to Qt5Agg if needed
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.gridspec import GridSpec
import warnings; warnings.filterwarnings('ignore')

# Try to import cartopy for natural borders; fall back gracefully
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ===========================================================================
# PART 1 — FILE I/O  (robust parser for mixed good/bad rows)
# ===========================================================================

def _parse_float(s):
    """Return float or NaN for any non-numeric token."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan


def load_reloc(f):
    """
    Load hypoDD.reloc or hypoDD.loc robustly.

    Handles rows where date/time fields contain '****', '-7', '-8' etc.
    (events that failed to relocate are flagged with cid < 0 or NaN coords).

    Columns expected:
      0:ID  1:LAT  2:LON  3:DEPTH  4:X(m)  5:Y(m)  6:Z(m)
      7:EX(m)  8:EY(m)  9:EZ(m)
      10:YR  11:MO  12:DY  13:HR  14:MI  15:SC
      16:MAG
      17:NCCP  18:NCCS  19:NCTP  20:NCTS
      21:RCC  22:RCT  23:CID
    """
    if not os.path.exists(f):
        raise FileNotFoundError(f"Cannot find: {f}")

    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Replace non-numeric date/time tokens with 0 so we get
            # a consistent column count. Tokens like '****', '-7' in
            # the time fields are harmless once we have the coords.
            tokens = line.split()
            if len(tokens) < 17:
                continue  # skip malformed lines
            # parse every token to float, using NaN for junk
            vals = [_parse_float(t) for t in tokens]
            # pad or trim to exactly 24 columns
            vals = vals[:24]
            while len(vals) < 24:
                vals.append(np.nan)
            rows.append(vals)

    if not rows:
        raise ValueError(f"No valid data rows found in {f}")

    d = np.array(rows, dtype=float)  # shape (n, 24)
    n = d.shape[0]

    def col(i):
        return d[:, i] if d.shape[1] > i else np.zeros(n)

    r = dict(
        cusp  = col(0),
        lat   = col(1),
        lon   = col(2),
        depth = col(3),          # km, positive down
        x     = col(4) / 1e3,   # m -> km
        y     = col(5) / 1e3,
        z     = -col(3),         # negative down for plotting
        ex    = col(7) / 1e3,
        ey    = col(8) / 1e3,
        ez    = col(9) / 1e3,
        mag   = col(16),
        nccp  = col(17),
        nccs  = col(18),
        nctp  = col(19),
        ncts  = col(20),
        rcc   = col(21),
        rct   = col(22),
        cid   = col(23),
    )

    # ------------------------------------------------------------------
    # Split into successfully relocated and failed events
    # A row is "good" if lat, lon and depth are all finite
    # ------------------------------------------------------------------
    good = (np.isfinite(r['lat']) &
            np.isfinite(r['lon']) &
            np.isfinite(r['depth']))

    n_good = good.sum()
    n_bad  = (~good).sum()
    print(f"  Rows read         : {n}")
    print(f"  Successfully relocated : {n_good}")
    print(f"  Failed / no solution   : {n_bad}")

    # Store the failed event IDs separately for diagnostics
    r['failed_cusp'] = r['cusp'][~good]

    # Keep only good rows for plotting
    for k in ('cusp','lat','lon','depth','x','y','z',
              'ex','ey','ez','mag','nccp','nccs',
              'nctp','ncts','rcc','rct','cid'):
        r[k] = r[k][good]

    # Fix zero magnitudes
    r['mag'][r['mag'] == 0] = 0.2

    # Fix zero error arrays (mirror MATLAB behaviour)
    if np.sum(np.abs(r['ex'])) == 0:
        r['ex'][:] = r['ey'][:] = r['ez'][:] = 1.0

    # Replace sentinel residual value -9.000 with NaN
    r['rcc'][r['rcc'] == -9.0] = np.nan
    r['rct'][r['rct'] == -9.0] = np.nan

    return r


def load_sta(f):
    """
    Load hypoDD.sta.
    Format per line: STA LAT LON ?? ?? NP NS NNP NNS ?? ?? ??
    Returns only stations with at least one observation.
    """
    if not f or not os.path.exists(f):
        return None
    rows = []
    with open(f) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 9 or p[0].startswith('#'):
                continue
            try:
                rows.append((p[0], float(p[1]), float(p[2]),
                             int(p[5]) + int(p[6]) + int(p[7]) + int(p[8])))
            except ValueError:
                continue
    if not rows:
        return None
    nm, la, lo, tot = zip(*rows)
    nm  = np.array(nm)
    la  = np.array(la,  dtype=float)
    lo  = np.array(lo,  dtype=float)
    tot = np.array(tot, dtype=int)
    m   = tot > 0
    return dict(name=nm[m], lat=la[m], lon=lo[m])


def load_lines(f):
    """Load geographic line file (lon lat rows; NaN NaN as separators)."""
    if f and os.path.exists(f):
        try:
            return np.loadtxt(f)
        except Exception:
            pass
    return None


def get_ne_borders():
    """
    Get Natural Earth state borders feature for cartopy.
    Returns cartopy feature object, or None if cartopy unavailable.
    """
    if not HAS_CARTOPY:
        return None
    try:
        return cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='10m', facecolor='none', edgecolor='none')
    except Exception:
        return None


def plot_ne_borders(ax):
    """
    Add Natural Earth borders to an axes.
    Uses a simple approach: draw US state borders from a default dataset.
    """
    if not HAS_CARTOPY:
        return
    try:
        borders = cfeature.NaturalEarthFeature(
            category='cultural', name='admin_1_states_provinces_lines',
            scale='10m', facecolor='none', edgecolor='#cccccc', linewidth=0.3)
        ax.add_feature(borders, transform=ccrs.PlateCarree())
    except Exception:
        pass


def print_summary(r, sta):
    print(f"\n{'─'*45}")
    print(f"  Events (relocated): {len(r['cusp'])}")
    print(f"  Failed events     : {len(r['failed_cusp'])}")
    print(f"  Mean EX/EY/EZ     : {r['ex'].mean():.3f} / "
          f"{r['ey'].mean():.3f} / {r['ez'].mean():.3f}  km")
    print(f"  Depth  min/max    : {r['depth'].min():.2f} / "
          f"{r['depth'].max():.2f}  km")
    print(f"  Mag    min/max    : {r['mag'].min():.1f} / "
          f"{r['mag'].max():.1f}")
    print(f"  Clusters          : {len(np.unique(r['cid']))}")
    if sta is not None:
        print(f"  Stations          : {len(sta['name'])}")
    print(f"{'─'*45}\n")


# ===========================================================================
# PART 2 — COLOURS
# ===========================================================================

# Colours
CE = '#d62728'   # events
CS = '#1f77b4'   # stations


# ===========================================================================
# PART 3 — DRAWING HELPERS
# ===========================================================================

def style_ax(ax, title, xlabel, ylabel):
    """Apply common axis styling."""
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


    # ===========================================================================
# PART 4 — SIMPLE STATIC FIGURE
# ===========================================================================

def plot_simple_comparison(reloc, loc, sta, border):
    """
    Create a simple static figure comparing initial and relocated positions.
    
    Parameters
    ----------
    reloc : dict from load_reloc() — relocated catalogue
    loc   : dict from load_reloc() — initial catalogue (optional)
    sta   : dict from load_sta() — stations (optional)
    border : array from load_lines() — geographic boundaries (optional)
    """
    if loc is None:
        # Single panel: just relocated events
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = [axes]
        titles = ['Relocated Events (.reloc)']
    else:
        # Two panels: before/after
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        titles = ['Initial Locations (.loc)', 'Relocated (.reloc)']
    
    fig.suptitle('HypoDD Event Locations', fontsize=12, fontweight='bold')
    
    catalogs = [loc, reloc] if loc is not None else [reloc]
    
    for idx, (ax, cat, title) in enumerate(zip(axes, catalogs, titles)):
        if cat is None:
            ax.set_visible(False)
            continue
        
        # Add Natural Earth borders if available
        if HAS_CARTOPY:
            try:
                borders_feat = cfeature.NaturalEarthFeature(
                    category='cultural', name='admin_1_states_provinces_lines',
                    scale='10m', facecolor='none', edgecolor='#e0e0e0', linewidth=0.5)
                ax.add_feature(borders_feat)
            except Exception:
                pass
        
        # Plot user-provided boundaries
        if border is not None:
            ax.plot(border[:, 0], border[:, 1],
                   lw=0.5, color='#999999', zorder=1, label='Boundary')
        
        # Plot events
        ax.plot(cat['lon'], cat['lat'], '.', ms=2,
                color=CE, alpha=0.6, zorder=2, label='Events')
        
        # Plot magnitude threshold circles
        big = cat['mag'] > 2.0
        if np.any(big):
            ax.plot(cat['lon'][big], cat['lat'][big], 'o', ms=6,
                   mfc='none', mec=CE, lw=0.8, zorder=3, label='M≥2.0')
        
        # Plot stations
        if sta is not None:
            ax.plot(sta['lon'], sta['lat'], '^', ms=6,
                   color=CS, zorder=4, label='Stations', alpha=0.7)
            for nm, lo, la in zip(sta['name'], sta['lon'], sta['lat']):
                ax.text(lo, la, nm, fontsize=4, va='bottom', ha='center',
                       zorder=5, alpha=0.5)
        
        # Set limits
        all_lon = cat['lon']
        all_lat = cat['lat']
        if sta is not None:
            all_lon = np.concatenate([all_lon, sta['lon']])
            all_lat = np.concatenate([all_lat, sta['lat']])
        
        plo = (all_lon.max() - all_lon.min()) * 0.05 + 0.01
        pla = (all_lat.max() - all_lat.min()) * 0.05 + 0.01
        ax.set_xlim(all_lon.min() - plo, all_lon.max() + plo)
        ax.set_ylim(all_lat.min() - pla, all_lat.max() + pla)
        ax.set_aspect('equal')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=9)
        ax.set_ylabel('Latitude', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    return fig

        # ===========================================================================
# PART 5 — DIAGNOSTIC PLOTS FIGURE
# ===========================================================================

class DiagFig:
    """
    Diagnostic figure for assessing HypoDD relocation quality.

    Panels
    ------
    Row 1: Error histograms (EX, EY, EZ)
    Row 2: Residual histograms (RCC, RCT) + pairs per event
    Row 3: Error vs depth, residual vs depth
    Row 4: Before/after comparison (if initial loc provided)
    """

    def __init__(self, reloc, loc=None):
        """
        Parameters
        ----------
        reloc : dict from load_reloc() — final relocated catalogue
        loc   : dict from load_reloc() — initial catalogue (optional)
        """
        self.r   = reloc
        self.loc = loc
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        nrows = 4 if self.loc is not None else 3
        self.fig, self.axes = plt.subplots(
            nrows, 3, figsize=(14, 4 * nrows))
        self.fig.canvas.manager.set_window_title(
            'HypoDD – Relocation Diagnostics')
        self.fig.subplots_adjust(
            left=0.07, right=0.97,
            top=0.94, bottom=0.06,
            hspace=0.50, wspace=0.32)
        self.fig.suptitle('HypoDD Relocation Diagnostics',
                          fontsize=11, fontweight='bold')
        self._row_errors()
        self._row_residuals()
        self._row_vs_depth()
        if self.loc is not None:
            self._row_before_after()
        self._hide_unused()

    # ------------------------------------------------------------------
    def _hist(self, ax, data, xlabel, color, title, units='km'):
        """Convenience histogram with mean/std annotation."""
        data = data[np.isfinite(data)]
        if len(data) == 0:
            ax.set_visible(False)
            return
        ax.hist(data, bins=40, color=color, edgecolor='white',
                linewidth=0.4, zorder=3)
        mu, sd = data.mean(), data.std()
        ax.axvline(mu, color='k', lw=1.2, ls='--', zorder=4)
        ax.text(0.97, 0.95,
                f'μ={mu:.3f} {units}\nσ={sd:.3f} {units}',
                transform=ax.transAxes,
                ha='right', va='top', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', alpha=0.8))
        style_ax(ax, title, xlabel, 'Count')

    def _scatter(self, ax, x, y, xlabel, ylabel, title,
                 color=CE, alpha=0.3, ms=2):
        """Convenience scatter with a horizontal zero line."""
        ax.plot(x, y, '.', ms=ms, color=color, alpha=alpha, zorder=3)
        ax.axhline(0, color='k', lw=0.8, ls='--', zorder=4)
        style_ax(ax, title, xlabel, ylabel)

    # ------------------------------------------------------------------
    def _row_errors(self):
        """Row 0: EX, EY, EZ histograms."""
        r = self.r
        pairs = [
            (r['ex'], 'EX (km)', '#4e79a7', 'Location Error X'),
            (r['ey'], 'EY (km)', '#f28e2b', 'Location Error Y'),
            (r['ez'], 'EZ (km)', '#e15759', 'Location Error Z'),
        ]
        for col, (data, xlabel, color, title) in enumerate(pairs):
            self._hist(self.axes[0, col], data, xlabel, color, title)

    # ------------------------------------------------------------------
    def _row_residuals(self):
        """Row 1: RCC histogram, RCT histogram, pairs-per-event bar."""
        r = self.r
        ax0, ax1, ax2 = self.axes[1]

        # RCC
        rcc = r['rcc'][np.isfinite(r['rcc'])]
        if rcc.any():
            self._hist(ax0, rcc, 'Mean CC residual (s)',
                       '#76b7b2', 'CC Residuals', units='s')

        # RCT
        rct = r['rct'][np.isfinite(r['rct'])]
        if rct.any():
            self._hist(ax1, rct, 'Mean CT residual (s)',
                       '#59a14f', 'CT Residuals', units='s')

        # Pairs per event
        total_pairs = r['nccp'] + r['nccs'] + r['nctp'] + r['ncts']
        ax2.hist(total_pairs, bins=40,
                 color='#edc948', edgecolor='white', linewidth=0.4)
        ax2.axvline(total_pairs.mean(), color='k', lw=1.2, ls='--')
        ax2.text(0.97, 0.95,
                 f'μ={total_pairs.mean():.1f}',
                 transform=ax2.transAxes,
                 ha='right', va='top', fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3',
                           fc='white', alpha=0.8))
        style_ax(ax2, 'Pairs per Event', 'Total pairs', 'Count')

    # ------------------------------------------------------------------
    def _row_vs_depth(self):
        """Row 2: EX vs depth, EZ vs depth, RCT vs depth."""
        r   = self.r
        dep = r['depth']   # km, positive down
        ax0, ax1, ax2 = self.axes[2]

        self._scatter(ax0, dep, r['ex'],
                      'Depth (km)', 'EX (km)',
                      'Horiz. Error vs Depth',
                      color='#4e79a7')
        # remove the meaningless zero line for error plots
        ax0.get_lines()[-1].set_visible(False)

        self._scatter(ax1, dep, r['ez'],
                      'Depth (km)', 'EZ (km)',
                      'Vert. Error vs Depth',
                      color='#e15759')
        ax1.get_lines()[-1].set_visible(False)

        rct = r['rct']
        finite = np.isfinite(rct)
        if np.any(finite):
            self._scatter(ax2, dep[finite], rct[finite],
                          'Depth (km)', 'RCT (s)',
                          'CT Residual vs Depth',
                          color='#59a14f')
        else:
            ax2.set_visible(False)

    # ------------------------------------------------------------------
    def _row_before_after(self):
        """
        Row 3: Before/after comparison of horizontal scatter,
        depth distribution, and error reduction.
        Only drawn when an initial loc file is supplied.
        """
        r   = self.r
        loc = self.loc
        ax0, ax1, ax2 = self.axes[3]

        # --- horizontal scatter before/after ---
        ax0.plot(loc['x'], loc['y'], '.', ms=2,
                 color='#aec7e8', alpha=0.5,
                 zorder=2, label='Initial')
        ax0.plot(r['x'], r['y'], '.', ms=2,
                 color=CE, alpha=0.7,
                 zorder=3, label='Relocated')
        ax0.set_aspect('equal')
        ax0.legend(fontsize=7, markerscale=3)
        style_ax(ax0, 'Before / After (Map)',
                 'Distance E–W (km)', 'Distance N–S (km)')

        # --- depth histograms before/after ---
        bins = np.linspace(
            min(loc['depth'].min(), r['depth'].min()),
            max(loc['depth'].max(), r['depth'].max()),
            40)
        ax1.hist(loc['depth'], bins=bins, orientation='horizontal',
                 color='#aec7e8', alpha=0.7,
                 edgecolor='white', lw=0.4, label='Initial')
        ax1.hist(r['depth'], bins=bins, orientation='horizontal',
                 color=CE, alpha=0.7,
                 edgecolor='white', lw=0.4, label='Relocated')
        ax1.invert_yaxis()
        ax1.legend(fontsize=7)
        style_ax(ax1, 'Depth Distribution',
                 'Count', 'Depth (km)')

        # --- error reduction ---
        # Use median per 0.5 km depth bin
        dep_bins = np.arange(
            r['depth'].min(),
            r['depth'].max() + 0.5, 0.5)
        mids = dep_bins[:-1] + 0.25

        def bin_median(vals, dep, bins):
            out = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (dep >= lo) & (dep < hi)
                out.append(np.median(vals[m]) if np.any(m) else np.nan)
            return np.array(out)

        ex_med  = bin_median(r['ex'],  r['depth'],  dep_bins)
        ey_med  = bin_median(r['ey'],  r['depth'],  dep_bins)
        ez_med  = bin_median(r['ez'],  r['depth'],  dep_bins)

        ax2.plot(ex_med, mids, '-o', ms=3, lw=1.0,
                 color='#4e79a7', label='EX')
        ax2.plot(ey_med, mids, '-o', ms=3, lw=1.0,
                 color='#f28e2b', label='EY')
        ax2.plot(ez_med, mids, '-o', ms=3, lw=1.0,
                 color='#e15759', label='EZ')
        ax2.invert_yaxis()
        ax2.legend(fontsize=7)
        style_ax(ax2, 'Median Error vs Depth',
                 'Error (km)', 'Depth (km)')

    # ------------------------------------------------------------------
    def _hide_unused(self):
        """Hide any axes left empty (e.g. if residuals are all zero)."""
        for row in self.axes:
            for ax in row:
                if not ax.get_visible():
                    ax.set_visible(False)


# ===========================================================================
# PART 5b — CLUSTER ANALYSIS FIGURE
# ===========================================================================

class ClusterFig:
    """
    Generate per-cluster summary figures showing cross-sections and statistics.
    """
    def __init__(self, reloc):
        """
        Parameters
        ----------
        reloc : dict from load_reloc() — relocated catalogue
        """
        self.r = reloc
        self.clusters = self._get_clusters()
        
    def _get_clusters(self):
        """Group events by cluster ID."""
        clusters = {}
        cids = np.unique(self.r['cid'])
        for cid in cids:
            if np.isfinite(cid):
                m = self.r['cid'] == cid
                clusters[int(cid)] = {
                    'cid': int(cid),
                    'n_events': m.sum(),
                    'lat': self.r['lat'][m],
                    'lon': self.r['lon'][m],
                    'depth': self.r['depth'][m],
                    'x': self.r['x'][m],
                    'y': self.r['y'][m],
                    'z': self.r['z'][m],
                    'mag': self.r['mag'][m],
                    'ex': self.r['ex'][m],
                    'ey': self.r['ey'][m],
                    'ez': self.r['ez'][m],
                }
        return clusters
    
    def plot_cluster_summaries(self):
        """Generate figure for each cluster showing cross-sections and map."""
        if not self.clusters:
            print("  No clusters found to visualize.")
            return
        
        for cid in sorted(self.clusters.keys()):
            self._plot_single_cluster(cid)
    
    def _plot_single_cluster(self, cid):
        """Create 3-panel figure for one cluster."""
        c = self.clusters[cid]
        
        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f'Cluster {cid} — {c["n_events"]} events',
                     fontsize=11, fontweight='bold')
        
        gs = fig.add_gridspec(1, 3, left=0.08, right=0.98, top=0.90, bottom=0.10)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_ns = fig.add_subplot(gs[0, 1])
        ax_ew = fig.add_subplot(gs[0, 2])
        
        # --- Map view ---
        ax_map.plot(c['x'], c['y'], '.', ms=3, color=CE, alpha=0.7)
        big = c['mag'] > 0.2
        if np.any(big):
            ax_map.plot(c['x'][big], c['y'][big], 'o', ms=8,
                        mfc='none', mec=CE, lw=1.0)
        # Add error bars
        for xi, yi, ei, fi in zip(c['x'], c['y'], c['ex'], c['ey']):
            ax_map.plot([xi - ei, xi + ei], [yi, yi],
                        color=CE, lw=0.5, alpha=0.3)
            ax_map.plot([xi, xi], [yi - fi, yi + fi],
                        color=CE, lw=0.5, alpha=0.3)
        ax_map.set_aspect('equal')
        ax_map.set_title('Map View', fontsize=9, fontweight='bold')
        ax_map.set_xlabel('E–W (km)', fontsize=8)
        ax_map.set_ylabel('N–S (km)', fontsize=8)
        ax_map.tick_params(labelsize=7)
        ax_map.grid(True, alpha=0.3)
        
        # --- N–S cross-section ---
        ax_ns.plot(c['x'], c['z'], '.', ms=3, color='#4e79a7', alpha=0.7)
        for xi, zi, ei, fi in zip(c['x'], c['z'], c['ex'], c['ez']):
            ax_ns.plot([xi - ei, xi + ei], [zi, zi],
                       color='#4e79a7', lw=0.5, alpha=0.3)
            ax_ns.plot([xi, xi], [zi - fi, zi + fi],
                       color='#4e79a7', lw=0.5, alpha=0.3)
        ax_ns.set_title('N–S Section', fontsize=9, fontweight='bold')
        ax_ns.set_xlabel('E–W (km)', fontsize=8)
        ax_ns.set_ylabel('Depth (km)', fontsize=8)
        ax_ns.invert_yaxis()
        ax_ns.tick_params(labelsize=7)
        ax_ns.grid(True, alpha=0.3)
        
        # --- E–W cross-section ---
        ax_ew.plot(c['y'], c['z'], '.', ms=3, color='#e15759', alpha=0.7)
        for yi, zi, fi, gi in zip(c['y'], c['z'], c['ey'], c['ez']):
            ax_ew.plot([yi - fi, yi + fi], [zi, zi],
                       color='#e15759', lw=0.5, alpha=0.3)
            ax_ew.plot([yi, yi], [zi - gi, zi + gi],
                       color='#e15759', lw=0.5, alpha=0.3)
        ax_ew.set_title('E–W Section', fontsize=9, fontweight='bold')
        ax_ew.set_xlabel('N–S (km)', fontsize=8)
        ax_ew.set_ylabel('Depth (km)', fontsize=8)
        ax_ew.invert_yaxis()
        ax_ew.tick_params(labelsize=7)
        ax_ew.grid(True, alpha=0.3)


                    # ===========================================================================
# PART 6 — ARGUMENT PARSER AND MAIN ENTRY POINT
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='HypoDD relocation comparison plotter + diagnostics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- files ---
    p.add_argument('--reloc',  default='hypoDD.reloc',
                   help='Relocated hypocenter file (hypoDD.reloc or hypoDD.loc)')
    p.add_argument('--loc',    default=None,
                   help='Initial hypocenter file for before/after comparison')
    p.add_argument('--sta',    default='hypoDD.sta',
                   help='Station file (hypoDD.sta)')
    p.add_argument('--line',   default=None,
                   help='Geographic line file (lon lat; NaN NaN separators)')

    # --- output ---
    p.add_argument('--nodiag', action='store_true',
                   help='Skip the diagnostics figure')
    p.add_argument('--clusters', action='store_true',
                   help='Generate per-cluster analysis figures (static images)')

    return p.parse_args()


def main():
    args = parse_args()

    # --- load data ---
    print(f"\n{'='*50}")
    print(' HypoDD Comparison Plotter')
    print(f"{'='*50}")

    try:
        reloc = load_reloc(args.reloc)
    except FileNotFoundError:
        print(f"ERROR: reloc file not found: {args.reloc}")
        print("       Use --reloc to specify the file path.")
        raise SystemExit(1)
    except ValueError as e:
        print(f"ERROR parsing reloc file: {e}")
        raise SystemExit(1)

    loc = None
    if args.loc:
        try:
            loc = load_reloc(args.loc)
            print(f"Initial locations loaded: {args.loc}")
        except Exception as e:
            print(f"WARNING: could not load --loc file: {e}")

    sta    = load_sta(args.sta)
    border = load_lines(args.line)

    print_summary(reloc, sta)

    # --- simple comparison figure ---
    fig = plot_simple_comparison(reloc=reloc, loc=loc, sta=sta, border=border)

    # --- diagnostic figure ---
    if not args.nodiag:
        diag_fig = DiagFig(reloc=reloc, loc=loc)

    # --- cluster analysis figures ---
    if args.clusters:
        print("\nGenerating per-cluster analysis figures...")
        cluster_fig = ClusterFig(reloc=reloc)
        cluster_fig.plot_cluster_summaries()
        print(f"  Generated {len(cluster_fig.clusters)} cluster figure(s)")

    plt.show()


if __name__ == '__main__':
    main()