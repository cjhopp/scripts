# CUSSP CASSM Site-Port Checklist (Inversion-focused)

This checklist ports assumptions from FSC MATLAB inversion logic to a new site where
changes are expected throughout the full array volume.

## 1) Geometry and indexing

- Receiver geometry source: use [python/dash_apps/cussp_seismicity_3d.py](python/dash_apps/cussp_seismicity_3d.py#L33) station CSV schema.
- Required receiver columns:
  - hmc_east_m
  - hmc_north_m
  - hmc_z_minus_depth_m (or hmc_z_m_asl fallback)
- Source geometry must be explicit CSV (source_id,x,y,z).
- Do not reuse FSC fixed 24x44 assumptions unless your source schedule actually matches.

## 2) Background model

- Set Vp background to 6900 m/s.
- Derive slowness from Vp directly for initialization.
- Recompute kernel G at this site from this site's geometry.

## 3) Prior mask

- FSC style localized prior mask is invalid here.
- Use full-domain prior mask (uniform weights) for all inversion cells inside the grid.
- If future data supports localization, add it as an optional, explicit mask file.

## 4) Rejection / QC assumptions

- Remove hard-coded bad-source IDs (e.g., FSC sources 18/23).
- Start with no static rejects and variance_reject_fraction=0.0.
- Add pair/source rejects only from new-site diagnostics.

## 5) Grid / regularization

- Build inversion bounds from source+receiver convex extent with margin.
- Keep 0.5 m spacing initially only if ray density supports it.
- Re-tune regularization per site (L-curve or holdout), do not copy FSC values blindly.

## 6) Picking and windows

- Rebuild travel-time picks and metric windows for this source wavelet/rock response.
- Validate window lengths against first-break stability and frequency content at new site.

## 7) Deliverables to produce before timelapse inversion

- sources_hmc.csv (source_id,x,y,z)
- stations_hmc.csv in seismicity-app schema
- pair geometry table (tx/rx per pair)
- full-domain prior mask
- baseline slowness model from Vp=6900 m/s

## 8) Files in this repo for this migration

- Settings: [python/vm/cussp_cassm_inversion_settings.yaml](python/vm/cussp_cassm_inversion_settings.yaml)
- Prep script: [python/vm/cussp_cassm_inversion_prep.py](python/vm/cussp_cassm_inversion_prep.py)
- Seismicity station schema reference: [python/dash_apps/cussp_seismicity_3d.py](python/dash_apps/cussp_seismicity_3d.py#L277)
