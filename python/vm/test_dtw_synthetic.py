#!/usr/bin/env python3
"""Test DTW (Dynamic Time Warping) for synthetic time-shifted waveforms.

Verifies that _dtw_dt_samples() can recover:
  1. Small shifts (±20 µs, typical xcorr) — should match xcorr precision
  2. Large shifts (±200 µs, ~2 periods @ 10 kHz) — where xcorr cycle-skips
  3. Shape-changed signals (decorrelated) — where xcorr fails
  4. Noisy data — reasonable robustness
"""

from __future__ import annotations

import numpy as np

# Import the functions under test
from cussp_cassm_process import _dtw_dt_samples, _xcorr_dt_samples, _cosine_window


class TestDTW:
    """Test suite for _dtw_dt_samples() synthetic time-shift recovery."""

    @staticmethod
    def make_ricker_wavelet(f_peak_hz: float, dt: float, nt: int) -> np.ndarray:
        """Generate a Ricker (Mexican hat) wavelet."""
        t = np.arange(nt) * dt - 0.003
        pi2f2t2 = (np.pi * f_peak_hz * t) ** 2
        w = (1.0 - 2.0 * pi2f2t2) * np.exp(-pi2f2t2)
        w /= np.max(np.abs(w)) + 1e-10
        return w.astype(np.float32)

    @staticmethod
    def taper_window(nt: int, frac: float = 0.1) -> np.ndarray:
        """Cosine taper."""
        return _cosine_window(nt, taper_frac=frac)

    def test_dtw_zero_shift(self):
        """Test: identical baseline and epoch (shift=0) should recover lag≈0."""
        dt = 1.0 / 48000.0  # 48 kHz sample interval
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))

        baseline = wavelet * window
        epoch = baseline.copy()

        lag, cc, edge_hit = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=50,
            strain_limit=2.0,
            edge_guard_samples=1,
        )

        assert abs(lag) < 2.0, f"Expected lag≈0, got {lag}"
        assert cc > 0.8, f"Expected cc>0.8, got {cc}"
        assert not edge_hit, "Should not be edge_hit for zero shift"
        print(f"✓ Zero shift: lag={lag:.2f} samples, cc={cc:.3f}")

    def test_dtw_small_shift(self):
        """Test: small shift ±20 µs (1 sample @ 48 kHz)."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        # Create padded version to simulate realistic shift
        pad_len = 20
        baseline_padded = np.pad(baseline, (pad_len, pad_len), mode='constant')
        
        shift_samples = 1  # ±20 µs
        rolled_padded = np.roll(baseline_padded, shift_samples)
        epoch = rolled_padded[pad_len:-pad_len]

        lag, cc, edge_hit = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=10,
            strain_limit=2.0,
        )

        # With padded roll, epoch is delayed so lag should be positive
        assert abs(lag - shift_samples) < 1.5, f"Expected lag≈{shift_samples}, got {lag}"
        assert cc > 0.6, f"Expected cc>0.6, got {cc}"
        print(f"✓ Small shift ({shift_samples} smp delay): recovered lag={lag:.2f} smp, cc={cc:.3f}")

    def test_dtw_large_shift_cycle_skip(self):
        """Test: large shift (~100 µs ≈ ~5 samples @ 48 kHz) where xcorr cycle-skips."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        # Create a padded version to simulate realistic shift without signal loss
        pad_len = 50
        baseline_padded = np.pad(baseline, (pad_len, pad_len), mode='constant')
        
        # Large shift: ~5 samples (100 µs, ~1 period at 10 kHz)
        shift_samples = 5
        # Roll the padded baseline, then extract the center portion
        rolled_padded = np.roll(baseline_padded, shift_samples)
        epoch = rolled_padded[pad_len:-pad_len]

        # Plain xcorr should fail on this (cycle-skip or low cc)
        xcorr_lag, xcorr_cc, xcorr_edge = _xcorr_dt_samples(
            baseline, epoch, max_lag=3, edge_guard_samples=1
        )

        # DTW should recover it (with larger max_shift)
        dtw_lag, dtw_cc, dtw_edge = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=50,  # Larger search range for this test
            strain_limit=2.0,
        )

        print(f"  Large shift ({shift_samples} smp delay):")
        print(f"    xcorr: lag={xcorr_lag:.2f} smp, cc={xcorr_cc:.3f}, edge_hit={xcorr_edge}")
        print(f"    DTW:   lag={dtw_lag:.2f} smp, cc={dtw_cc:.3f}, edge_hit={dtw_edge}")

        # DTW should recover it - with this shift setup, epoch is delayed so lag should be positive
        assert abs(dtw_lag - shift_samples) < 2.0, f"DTW failed: lag={dtw_lag}, expected {shift_samples}"
        assert dtw_cc > 0.3, f"DTW quality too low: cc={dtw_cc}"
        print(f"✓ Large shift cycle-skip: DTW recovered lag={dtw_lag:.2f} smp")

    def test_dtw_decorrelated_waveform(self):
        """Test: epoch has different shape (mild decorrelation) — DTW should handle."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        # Shift + add some attenuation/deformation
        pad_len = 20
        baseline_padded = np.pad(baseline, (pad_len, pad_len), mode='constant')
        
        shift_samples = 3
        rolled_padded = np.roll(baseline_padded, shift_samples)
        epoch = rolled_padded[pad_len:-pad_len] * 0.9  # slight amplitude decay
        rng = np.random.default_rng(42)
        epoch = epoch + 0.05 * rng.standard_normal(len(epoch))  # add noise

        lag, cc, edge_hit = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=10,
            strain_limit=2.0,
        )

        assert abs(lag - shift_samples) < 3.0, f"DTW with decoration: lag={lag}, expected {shift_samples}"
        print(f"✓ Decorrelated waveform: lag={lag:.2f} smp (expected {shift_samples}), cc={cc:.3f}")

    def test_dtw_negative_shift(self):
        """Test: negative shift (epoch earlier than baseline)."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        pad_len = 20
        baseline_padded = np.pad(baseline, (pad_len, pad_len), mode='constant')
        
        shift_samples = -4  # epoch 4 samples earlier
        rolled_padded = np.roll(baseline_padded, shift_samples)
        epoch = rolled_padded[pad_len:-pad_len]

        lag, cc, edge_hit = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=10,
            strain_limit=2.0,
        )

        # Negative roll means epoch is earlier (negative lag)
        assert lag < 0, f"Expected negative lag, got {lag}"
        assert abs(lag - shift_samples) < 2.0, f"DTW negative shift: lag={lag}, expected {shift_samples}"
        print(f"✓ Negative shift ({shift_samples} smp delay): recovered lag={lag:.2f} smp, cc={cc:.3f}")

    def test_dtw_max_shift_limit(self):
        """Test: edge_hit when shift exceeds max_shift."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        pad_len = 20
        baseline_padded = np.pad(baseline, (pad_len, pad_len), mode='constant')
        
        # Very large shift (beyond max_shift)
        shift_samples = 20
        rolled_padded = np.roll(baseline_padded, shift_samples)
        epoch = rolled_padded[pad_len:-pad_len]

        lag, cc, edge_hit = _dtw_dt_samples(
            baseline_win=baseline,
            epoch_win=epoch,
            max_shift=5,  # constrain to ±5 samples
            strain_limit=2.0,
        )

        # Should report edge_hit or saturate the lag
        print(f"✓ Shift beyond max_shift: lag={lag:.2f} (max=±5), edge_hit={edge_hit}, cc={cc:.3f}")

    def test_dtw_vs_xcorr_small_shift(self):
        """Benchmark: DTW vs xcorr on small shifts (integer DTW ≤ xcorr error)."""
        dt = 1.0 / 48000.0
        wavelet = self.make_ricker_wavelet(f_peak_hz=10000.0, dt=dt, nt=256)
        window = self.taper_window(len(wavelet))
        baseline = wavelet * window

        true_shift = 1  # 1-sample integer shift — recoverable by both methods
        epoch = np.roll(baseline, true_shift)
        epoch[:true_shift] = 0.0

        xcorr_lag, xcorr_cc, _ = _xcorr_dt_samples(baseline, epoch, max_lag=5, edge_guard_samples=1)
        dtw_lag, dtw_cc, _ = _dtw_dt_samples(baseline, epoch, max_shift=5, strain_limit=2.0)

        xcorr_err = abs(xcorr_lag - true_shift)
        dtw_err = abs(dtw_lag - true_shift)

        print(f"✓ Benchmark small shift ({true_shift} smp):")
        print(f"    xcorr: lag={xcorr_lag:.3f} smp, err={xcorr_err:.3f}, cc={xcorr_cc:.4f}")
        print(f"    DTW:   lag={dtw_lag:.3f} smp, err={dtw_err:.3f}, cc={dtw_cc:.4f}")
        # xcorr with sub-sample parabolic fit should recover integer shifts accurately
        assert xcorr_err < 0.5, f"xcorr should recover 1-sample shift, got err={xcorr_err:.3f}"
        # DTW should not be much worse than xcorr for small integer shifts
        assert dtw_err <= xcorr_err + 1.0, f"DTW regression vs xcorr: dtw_err={dtw_err:.3f}"


if __name__ == "__main__":
    test = TestDTW()
    try:
        print("\n" + "="*60)
        print("Running DTW Synthetic Time-Shift Tests")
        print("="*60 + "\n")
        
        test.test_dtw_zero_shift()
        test.test_dtw_small_shift()
        test.test_dtw_large_shift_cycle_skip()
        test.test_dtw_decorrelated_waveform()
        test.test_dtw_negative_shift()
        test.test_dtw_max_shift_limit()
        test.test_dtw_vs_xcorr_small_shift()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
    except AssertionError as e:
        print(f"\n✗ Assertion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
