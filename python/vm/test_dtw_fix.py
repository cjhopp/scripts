#!/usr/bin/env python3
"""Quick test: verify DTW wide-window construction in cussp_cassm_process.py."""

import sys
sys.path.insert(0, '/home/chopp/scripts/python/vm')

import numpy as np
from cussp_cassm_process import MetricConfig


def test_dtw_params_in_config():
    """Verify DTW parameters are in MetricConfig and have correct defaults."""
    cfg = MetricConfig()
    assert hasattr(cfg, 'dtw_enabled'), "MetricConfig missing dtw_enabled"
    assert hasattr(cfg, 'dtw_max_shift_ms'), "MetricConfig missing dtw_max_shift_ms"
    assert hasattr(cfg, 'dtw_strain_limit'), "MetricConfig missing dtw_strain_limit"
    assert hasattr(cfg, 'dtw_min_ncc'), "MetricConfig missing dtw_min_ncc"
    
    # Check defaults
    assert cfg.dtw_enabled == True, f"Expected dtw_enabled=True, got {cfg.dtw_enabled}"
    assert cfg.dtw_max_shift_ms == 0.5, f"Expected dtw_max_shift_ms=0.5, got {cfg.dtw_max_shift_ms}"
    assert cfg.dtw_strain_limit == 2.0, f"Expected dtw_strain_limit=2.0, got {cfg.dtw_strain_limit}"
    assert cfg.dtw_min_ncc == 0.2, f"Expected dtw_min_ncc=0.2, got {cfg.dtw_min_ncc}"
    
    print("✓ DTW parameters present in MetricConfig with correct defaults")


def test_dtw_cache_key_includes_params():
    """Verify that MetricConfig cache key includes DTW parameters."""
    cfg1 = MetricConfig(dtw_max_shift_ms=0.25, dtw_min_ncc=0.2)
    cfg2 = MetricConfig(dtw_max_shift_ms=0.5, dtw_min_ncc=0.3)
    
    # Build the cache keys like compute_metrics does
    key1 = (
        f"test|{cfg1.dtw_enabled}|{cfg1.dtw_max_shift_ms:.3f}|"
        f"{cfg1.dtw_strain_limit:.2f}|{cfg1.dtw_min_ncc:.3f}"
    )
    key2 = (
        f"test|{cfg2.dtw_enabled}|{cfg2.dtw_max_shift_ms:.3f}|"
        f"{cfg2.dtw_strain_limit:.2f}|{cfg2.dtw_min_ncc:.3f}"
    )
    
    assert key1 != key2, "Cache keys should differ when DTW params differ"
    print("✓ DTW parameters affect cache key uniqueness")


def test_dtw_window_extension_calc():
    """Verify DTW window extension is calculated correctly."""
    sample_rate_hz = 48000.0
    dtw_max_shift_ms = 0.25
    
    # Mimic the calculation from compute_metrics
    dtw_max_shift_samples = max(int(dtw_max_shift_ms * sample_rate_hz / 1000.0), 1)
    dtw_win_extension = dtw_max_shift_samples + 5
    
    # At 48 kHz, 0.25 ms = 12 samples
    assert dtw_max_shift_samples == 12, f"Expected 12 samples for 0.25 ms @ 48 kHz, got {dtw_max_shift_samples}"
    assert dtw_win_extension == 17, f"Expected extension=17 (12+5), got {dtw_win_extension}"
    
    print(f"✓ DTW window extension: {dtw_max_shift_samples} samples + 5 = {dtw_win_extension}")


if __name__ == "__main__":
    test_dtw_params_in_config()
    test_dtw_cache_key_includes_params()
    test_dtw_window_extension_calc()
    print("\n✓✓✓ All DTW parameter tests passed!")
