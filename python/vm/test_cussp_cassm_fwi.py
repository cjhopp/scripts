from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from cussp_cassm_fwi import (
    AnalyticSolver,
    FD2DSolver,
    FWIGrid,
    _ricker_wavelet,
    build_fwi_context,
    estimate_source_wavelet,
    fwi_estimate_dt,
)


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_build_fwi_context_uses_csv_ids(tmp_path: Path) -> None:
    sources_csv = _write_text(
        tmp_path / "sources_hmc.csv",
        """
source_id,x,y,z,amplifier_channel,borehole,depth_m,source_index
AMLS1,1227.5331395080257,-865.1683757559974,333.5934906984,1,AML,14.518,1
AMLS2,1237.8932440343551,-866.7880770737479,324.6802121064,2,AML,28.31,2
AMLS3,1248.250796341293,-868.2762929927587,315.787607184,3,AML,42.072,3
AMLS4,1258.5911769662098,-869.7971476664018,306.95123268000003,4,AML,55.788,4
AMUS1,1230.8987156692172,-865.3992920751302,341.4457882272,5,AMU,14.435,1
AMUS2,1244.2400801409058,-868.0718059542567,339.6244247376,6,AMU,28.212,2
AMUS3,1257.6023869771948,-870.7668233712961,338.0027893728,7,AMU,41.989,3
AMUS4,1270.9565239606025,-873.4888698960982,336.63756456960004,8,AMU,55.736,4
DMLS1,1216.409833968907,-878.8265846066049,335.3401467936,9,DML,13.862,1
DMLS2,1224.6894182185536,-884.284779576907,328.1827725384,10,DML,26.115,2
DMLS3,1232.9482372033342,-889.6490940500196,321.0342643056,11,DML,38.307,3
DMLS4,1241.2503234754236,-894.9896714246374,313.8664607088,12,DML,50.53,4
DMUS1,1219.3444657889625,-878.8233624700633,343.2270708216,13,DMU,13.945,1
DMUS2,1230.191167807151,-884.3001139997479,342.97920929040004,14,DMU,26.137,2
DMUS3,1241.1245768875192,-889.8127988928958,342.960050172,15,DMU,38.42,3
DMUS4,1251.9994106399558,-895.3025447353208,343.175283168,16,DMU,50.642,4
""",
    )
    receivers_csv = _write_text(
        tmp_path / "receivers_hmc.csv",
        """
receiver_id,x,y,z
AML1,1233.9997361617309,-866.2194598412806,328.034542812
AML2,1243.1798843695012,-867.5346245032979,320.1368191104
AML3,1252.407480438013,-868.8871660916373,312.2303622792
AML4,1261.6471564701933,-870.214070530272,304.35762766656
AMU1,1239.191149976464,-867.0579539564026,340.29718935600005
AMU2,1251.050957508334,-869.4447024870778,338.767596276
AMU3,1262.944983262825,-871.8519945386215,337.4282352768
AMU4,1274.8387215933424,-874.2831275883755,336.282194592
CEnc,1229.4254391694594,-890.5480899999158,330.0
CMon,1229.4254391694594,-890.5480899999158,330.0
CTrig,1229.4254391694594,-890.5480899999158,330.0
DML1,1219.1079334126284,-880.6079502323437,333.00986057280005
DML2,1227.3638597157576,-886.0247480470543,325.8716628096
DML3,1235.6231863960718,-891.3826896964263,318.71895108000007
DML4,1243.9728161025423,-896.7399121091172,311.5150768416
DMU1,1222.8684139137054,-880.6045885932699,343.12201571040004
DMU2,1233.7722435997853,-886.1057810413636,342.946712124
DMU3,1244.6233549583012,-891.578987327568,343.0035500088
DMU4,1255.470191714659,-897.0544217060431,343.2762198216
PPS,1229.4254391694594,-890.5480899999158,330.0
TS01,1225.3516984061453,-902.4552329048828,336.9431082816
TS02,1227.322688389254,-901.0970611949584,336.28012530480004
TS03,1229.295795826544,-899.7379383375,335.6254920192
TS04,1231.2903475697562,-898.3645576999119,334.97292924
TS05,1233.2747298075735,-896.9984831560803,334.33242831120003
TS06,1235.2632836043804,-895.6296661992321,333.69896247120005
TS07,1237.2535693603893,-894.25978651475,333.0733242
TS08,1239.2455658869185,-892.8888596140448,332.45551776480005
TS09,1241.2489758781887,-891.510636850631,331.8416587944
TS10,1243.2492236336032,-890.1354966502433,331.235474688
TS11,1245.241417261368,-888.7667923753123,330.6383748408
TS12,1247.2328270399369,-887.3995220276622,330.04812354480003
TS13,1249.233650990725,-886.0271180158516,329.4632117352
TS14,1251.2515926282813,-884.64466875382,328.88273080320005
TS15,1253.2573453073005,-883.2722512480993,328.315111212
TS16,1255.265510836186,-881.8998496623375,327.756112584
TS17,1257.2799302447083,-880.5230979244011,327.20320660320004
TS18,1259.285102510028,-879.1512509250107,326.6594766264
TS19,1261.315294105962,-877.7608486202747,326.11570397760005
TS20,1263.3264148194994,-876.3820879176317,325.5837273936
TS21,1265.3484299676088,-874.9954383363824,325.05642583200006
TS22,1267.359689663597,-873.6164781590626,324.54005256960005
TS23,1269.3699682805047,-872.2385020583924,324.0320085768
TS24,1271.6465101700444,-870.6782632335983,323.4631670424
TSS1,1226.4177469054237,-901.7205762506088,336.58339581840005
TSS2,1241.9074373306198,-891.0578545696718,331.64137014240004
TSS3,1256.6532713945692,-880.9515395799305,327.3744917064
TSS4,1270.6125931445124,-871.3868590148816,323.721410976
""",
    )

    source_boreholes = [
        "AML", "AML", "AML", "AML",
        "AMU", "AMU", "AMU", "AMU",
        "DML", "DML", "DML", "DML",
        "DMU", "DMU", "DMU", "DMU",
    ]
    sample_rate_hz = 48000.0
    sample_count = 1200
    n_sources = len(source_boreholes)
    n_receivers = 72
    d_obs_baseline = np.zeros((n_sources * n_receivers, sample_count), dtype=np.float32)
    baseline_picks = np.zeros(n_sources * n_receivers, dtype=np.int32)

    ctx = build_fwi_context(
        tg_n_sources=n_sources,
        tg_n_receivers=n_receivers,
        tg_sample_rate_hz=sample_rate_hz,
        tg_sample_count=sample_count,
        d_obs_baseline=d_obs_baseline,
        baseline_picks=baseline_picks,
        source_boreholes=source_boreholes,
        sources_csv=sources_csv,
        receivers_csv=receivers_csv,
        solver_name="analytic",
        grid_dx_m=0.5,
        grid_dz_m=0.5,
        grid_padding_m=10.0,
        vp_background_mps=3000.0,
        dt_search_max_ms=2.0,
        min_ncc=0.2,
    )

    assert ctx.src_global_idx.tolist() == [8, 9, 10, 11, 12, 13, 14, 15]
    assert ctx.rec_global_idx.tolist() == list(range(48, 72))
    assert ctx.source_wavelets.shape == (8, ctx.grid.nt)
    assert ctx.rec_pos_grid.shape[0] == 24
    assert ctx.grid.nt >= 10


def test_build_fwi_context_resamples_source_wavelets_to_grid_axis(tmp_path: Path) -> None:
    sources_csv = _write_text(
        tmp_path / "sources_hmc.csv",
        """
source_id,x,y,z,amplifier_channel,borehole,depth_m,source_index
DMX1,0.0,0.0,0.0,1,DMX,10.0,1
""",
    )
    receivers_csv = _write_text(
        tmp_path / "receivers_hmc.csv",
        """
receiver_id,x,y,z
TS01,12.0,0.0,-2.0
""",
    )

    sample_rate_hz = 1000.0
    sample_count = 20
    n_sources = 1
    n_receivers = 49
    d_obs_baseline = np.zeros((n_sources * n_receivers, sample_count), dtype=np.float32)
    baseline_picks = np.zeros(n_sources * n_receivers, dtype=np.int32)
    d_obs_baseline[48, :] = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    baseline_picks[48] = 0

    gate_pre_ms = 1.0
    gate_post_ms = 18.0

    ctx = build_fwi_context(
        tg_n_sources=n_sources,
        tg_n_receivers=n_receivers,
        tg_sample_rate_hz=sample_rate_hz,
        tg_sample_count=sample_count,
        d_obs_baseline=d_obs_baseline,
        baseline_picks=baseline_picks,
        source_boreholes=["DMX"],
        sources_csv=sources_csv,
        receivers_csv=receivers_csv,
        solver_name="analytic",
        grid_dx_m=0.5,
        grid_dz_m=0.5,
        grid_padding_m=0.5,
        vp_background_mps=6900.0,
        gate_pre_ms=gate_pre_ms,
        gate_post_ms=gate_post_ms,
    )

    raw_wavelets = estimate_source_wavelet(
        d_obs_baseline=d_obs_baseline,
        baseline_picks=baseline_picks,
        gate_pre_samples=max(int(gate_pre_ms * sample_rate_hz / 1000.0), 1),
        gate_post_samples=max(int(gate_post_ms * sample_rate_hz / 1000.0), 1),
        src_indices=np.array([0], dtype=np.int32),
        rec_indices=np.array([48], dtype=np.int32),
        n_receivers=n_receivers,
    )
    expected = np.interp(
        np.arange(ctx.grid.nt, dtype=np.float64) * ctx.grid.dt,
        np.arange(raw_wavelets.shape[1], dtype=np.float64) / sample_rate_hz,
        raw_wavelets[0].astype(np.float64),
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    assert ctx.source_wavelets.shape == (1, ctx.grid.nt)
    np.testing.assert_allclose(ctx.source_wavelets[0], expected, rtol=1e-6, atol=1e-6)


def test_analytic_round_trip_and_fd2d_peak_time_agree() -> None:
    grid = FWIGrid(
        nx=128,
        nz=128,
        dx=0.05,
        dz=0.05,
        x0=0.0,
        z0=0.0,
        dt=5.0e-6,
        nt=1200,
        x=np.arange(128, dtype=np.float64) * 0.05,
        z=np.arange(128, dtype=np.float64) * 0.05,
    )
    vp = np.full((grid.nz, grid.nx), 3000.0, dtype=np.float64)
    source_wavelet = _ricker_wavelet(5000.0, grid.dt, grid.nt).astype(np.float64)
    fd_wavelet_peak_hz = 2000.0
    fd_source_wavelet = _ricker_wavelet(fd_wavelet_peak_hz, grid.dt, grid.nt).astype(np.float64)
    src_ix = 16
    src_iz = 16
    rec_ix = np.array([116], dtype=np.int32)
    rec_iz = np.array([16], dtype=np.int32)

    analytic_solver = AnalyticSolver()
    d_obs = analytic_solver.forward(
        vp=vp,
        source_wavelet=source_wavelet,
        src_ix=src_ix,
        src_iz=src_iz,
        rec_ix=rec_ix,
        rec_iz=rec_iz,
        grid=grid,
    )[0]

    dt_us, ncc, rejected = fwi_estimate_dt(
        bl_win=d_obs,
        ep_win=d_obs,
        vp=vp,
        grid=grid,
        source_wavelet=source_wavelet,
        src_ix=src_ix,
        src_iz=src_iz,
        rec_ix=int(rec_ix[0]),
        rec_iz=int(rec_iz[0]),
        solver=analytic_solver,
        freq_bands=[(1000.0, 8000.0)],
        sample_rate_hz=1.0 / grid.dt,
        dt_search_max_s=5.0e-4,
        min_ncc=0.8,
    )

    fd_solver = FD2DSolver(cpml_thickness=4)
    d_obs_fd = analytic_solver.forward(
        vp=vp,
        source_wavelet=fd_source_wavelet,
        src_ix=src_ix,
        src_iz=src_iz,
        rec_ix=rec_ix,
        rec_iz=rec_iz,
        grid=grid,
    )[0]
    d_fd = fd_solver.forward(
        vp=vp,
        source_wavelet=fd_source_wavelet,
        src_ix=src_ix,
        src_iz=src_iz,
        rec_ix=rec_ix,
        rec_iz=rec_iz,
        grid=grid,
    )[0]

    distance_m = float(np.hypot((rec_ix[0] - src_ix) * grid.dx, (rec_iz[0] - src_iz) * grid.dz))
    expected_peak = int(round((distance_m / 3000.0) / grid.dt))
    search_start = max(expected_peak - 80, 0)
    search_stop = min(expected_peak + 80, grid.nt)
    analytic_peak = search_start + int(np.argmax(np.abs(d_obs_fd[search_start:search_stop])))
    fd_peak = search_start + int(np.argmax(np.abs(d_fd[search_start:search_stop])))
    peak_tolerance = max(2, int(round(1.0 / (fd_wavelet_peak_hz * grid.dt))))

    assert abs(dt_us) <= 1.0
    assert ncc > 0.8
    assert not rejected
    assert abs(fd_peak - analytic_peak) <= peak_tolerance
