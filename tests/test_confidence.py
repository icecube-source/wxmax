"""Tests for the calibrated P(peak passed) confidence + lock guards."""
from wxmax.nowcast.confidence import (
    decide_lock,
    p_model,
    p_traj,
    peak_passed_confidence,
    real_time_best,
)


def test_p_traj_monotone_and_limits():
    # short flat (Miami 1:30pm: ~18 min, 0.4°F drop) -> low
    assert p_traj(18, 0.4) < 0.1
    # long sustained decline -> ~1
    assert p_traj(180, 3.0) > 0.99
    assert p_traj(180, 3.0) > p_traj(60, 1.0) > p_traj(18, 0.4)


def test_p_model_limits():
    assert p_model(0.0) > 0.99          # no remaining rise -> ~1
    assert p_model(2.0) < 0.1           # 2°F more expected -> ~0
    assert p_model(0.0) > p_model(1.0) > p_model(2.0)


def test_confidence_none_passthrough():
    assert peak_passed_confidence(None, 180, 3.0, 0.0) is None
    assert decide_lock(None, 18.0, 16.0, 90.0, 0.0) is False


def test_strong_past_peak_state_locks():
    conf = peak_passed_confidence(p_clim=1.0, dwell_min=150, drop_f=3.0, remaining_rise=0.0)
    assert conf >= 0.99
    assert decide_lock(conf, now_hour=17.5, p_lock=16.0, dwell_min=150, remaining_rise=0.0) is True


def test_guards_block_premature_locks():
    strong = peak_passed_confidence(1.0, 150, 3.0, 0.0)
    # before the safe hour -> no lock even if confidence is high
    assert decide_lock(strong, now_hour=13.5, p_lock=16.0, dwell_min=150, remaining_rise=0.0) is False
    # model still expects a rise -> guard blocks
    assert decide_lock(strong, now_hour=17.5, p_lock=16.0, dwell_min=150, remaining_rise=1.0) is False
    # too-short dwell -> guard blocks
    assert decide_lock(strong, now_hour=17.5, p_lock=16.0, dwell_min=30, remaining_rise=0.0) is False


def test_decide_lock_threshold_default_is_085():
    assert decide_lock(0.86, now_hour=17, p_lock=13, dwell_min=60, remaining_rise=0.0) is True
    assert decide_lock(0.80, now_hour=17, p_lock=13, dwell_min=60, remaining_rise=0.0) is False


def test_real_time_best_converges_and_floors():
    # P=0 -> forward forecast (max of obs_max, obs_now+rise, morning_est)
    b0, lo0, hi0 = real_time_best(80, 78, 4.0, morning_est=85, morning_half=4, p_passed=0.0, locked=False)
    assert b0 == 85 and lo0 >= 80 - 1e-9            # forward=85, lower bound >= obs_max
    # P=1 -> realized max
    b1, _, _ = real_time_best(80, 78, 4.0, 85, 4, p_passed=1.0, locked=False)
    assert abs(b1 - 80) < 1e-9
    # locked -> realized max, tight band
    bl, lol, hil = real_time_best(80, 76, 0.0, 85, 4, p_passed=0.9, locked=True)
    assert bl == 80 and lol == 80 and abs(hil - 80.7) < 1e-9


def test_miami_1330_case_does_not_lock():
    # P_clim(13.5)~0.37 (measured), 18-min flat, 0.4°F drop -> confidence far below 0.99
    conf = peak_passed_confidence(p_clim=0.37, dwell_min=18, drop_f=0.4, remaining_rise=0.0)
    assert conf < 0.5
    assert decide_lock(conf, now_hour=13.5, p_lock=16.9, dwell_min=18, remaining_rise=0.0) is False
