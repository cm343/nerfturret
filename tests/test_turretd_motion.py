"""Tests for src/turretd/motion.py — axis kinematics and the worker loop (sim pins)."""

import time

import pytest

from turretd.motion import AxisKinematics, AxisPins, MotionClient

DT = 0.0005          # 2 kHz simulation tick


def run(ax: AxisKinematics, seconds: float, dt: float = DT) -> list[int]:
    """Advance the axis; return every emitted step count."""
    out = []
    for _ in range(int(seconds / dt)):
        out.append(ax.update(dt))
    return out


# ── velocity mode ────────────────────────────────────────────────────────────
def test_velocity_ramps_up_then_cruises():
    ax = AxisKinematics(accel=1000)
    ax.velocity(500)
    run(ax, 0.25)                         # 0.25 s × 1000 = 250 < 500: still ramping
    assert 200 < ax.v < 300
    run(ax, 0.5)
    assert ax.v == pytest.approx(500)
    before = ax.pos
    run(ax, 1.0)
    assert ax.pos - before == pytest.approx(500, abs=2)


def test_velocity_zero_decelerates_and_goes_idle():
    ax = AxisKinematics(accel=1000)
    ax.velocity(500)
    run(ax, 1.0)
    ax.velocity(0)
    run(ax, 0.2)
    assert ax.moving                      # 500/1000 = 0.5 s to stop
    run(ax, 0.4)
    assert not ax.moving and ax.v == 0


def test_negative_velocity_moves_backwards():
    ax = AxisKinematics(accel=2000)
    ax.velocity(-400)
    steps = run(ax, 1.0)
    assert all(s <= 0 for s in steps)
    assert ax.pos < -250


def test_velocity_stops_exactly_at_limit():
    ax = AxisKinematics(accel=2000)
    ax.set_limits(-100, 300)
    ax.velocity(1000)
    steps = run(ax, 3.0)
    assert ax.pos == 300 and not ax.moving
    assert all(s >= 0 for s in steps)     # no overshoot-and-return
    ax.velocity(1000)                     # pushing further does nothing
    run(ax, 0.5)
    assert ax.pos == 300
    ax.velocity(-1000)
    run(ax, 3.0)
    assert ax.pos == -100


# ── position mode ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target", [1, 7, 250, -1234, 5000])
def test_goto_lands_exactly_without_overshoot(target):
    ax = AxisKinematics(accel=3000)
    ax.goto(target, 2000)
    steps = run(ax, 6.0)
    assert ax.pos == target and not ax.moving
    sign = 1 if target > 0 else -1
    assert all(s * sign >= 0 for s in steps)


def test_goto_is_clamped_to_limits():
    ax = AxisKinematics(accel=3000)
    ax.set_limits(0, 100)
    ax.goto(500, 1000)
    run(ax, 2.0)
    assert ax.pos == 100


def test_goto_respects_vmax():
    ax = AxisKinematics(accel=100000)
    ax.goto(10000, 800)
    run(ax, 1.0)
    assert abs(ax.v) <= 800 + 1e-6


def test_velocity_zero_during_goto_stops_with_ramp():
    ax = AxisKinematics(accel=1000)
    ax.goto(100000, 1000)
    run(ax, 2.0)                          # cruising at 1000
    ax.velocity(0)
    p = ax.pos
    run(ax, 2.0)
    assert not ax.moving
    assert 400 <= ax.pos - p <= 600       # v²/2a = 500 steps to stop


def test_halt_is_instant():
    ax = AxisKinematics(accel=1000)
    ax.velocity(500)
    run(ax, 1.0)
    ax.halt()
    p = ax.pos
    run(ax, 0.5)
    assert ax.pos == p and ax.v == 0


def test_set_position_rezeroes():
    ax = AxisKinematics()
    ax.goto(50, 1000)
    run(ax, 1.0)
    ax.set_position(0)
    assert ax.pos == 0 and not ax.moving


# ── worker loop with sim pins (thread) ───────────────────────────────────────
PINS = [AxisPins(16, 20, 6), AxisPins(23, 24, 5), AxisPins(17, 27, 4)]


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def client():
    c = MotionClient(PINS, 'sim', use_process=False)
    c.start()
    yield c
    c.stop()


def test_worker_goto_pulses_and_reports(client):
    client.send('params', 2, {'accel': 20000})
    client.send('goto', 2, 400, 4000)
    assert wait_until(lambda: client.axis(2).pos == 400 and not client.axis(2).moving)
    pins = client.worker.pins
    assert pins.pulses[17] == 400               # STEP pin of axis 2
    assert pins.level[4] == 0                   # EN low = enabled
    assert client.axis(2).enabled


def test_worker_moves_two_axes_at_once(client):
    for i in (0, 1):
        client.send('params', i, {'accel': 20000})
    client.send('goto', 0, 300, 3000)
    client.send('goto', 1, -300, 3000)
    assert wait_until(lambda: client.axis(0).pos == 300 and client.axis(1).pos == -300)
    assert client.worker.pins.level[24] == 0    # DIR low for negative moves


def test_worker_idle_disable(client):
    client.send('params', 2, {'accel': 20000, 'hold': False, 'idle_disable': 0.05})
    client.send('move_by', 2, 50, 4000)
    assert wait_until(lambda: client.axis(2).pos == 50)
    assert wait_until(lambda: not client.axis(2).enabled)
    assert client.worker.pins.level[4] == 1     # EN high = disabled


def test_worker_limits_and_raw_move(client):
    client.send('params', 0, {'accel': 20000})
    client.send('limits', 0, -50, 50)
    client.send('move_by', 0, 500, 4000)
    assert wait_until(lambda: not client.axis(0).moving and client.axis(0).pos == 50)
    assert (client.axis(0).lo, client.axis(0).hi) == (-50, 50)
    client.send('raw_move_by', 0, 100, 4000)    # diagnostics ignore limits
    assert wait_until(lambda: client.axis(0).pos == 150)


def test_worker_halt_all_and_shutdown_disables(client):
    client.send('params', 0, {'accel': 1000})
    client.send('vel', 0, 2000)
    assert wait_until(lambda: client.axis(0).moving)
    client.send('halt_all')
    assert wait_until(lambda: not client.axis(0).moving)
    client.stop()
    assert client.worker.pins.level[6] == 1     # EN released on shutdown


def test_worker_runs_in_a_process():
    c = MotionClient(PINS, 'sim', use_process=True)
    c.start()
    try:
        c.send('params', 1, {'accel': 20000})
        c.send('goto', 1, 200, 4000)
        assert wait_until(lambda: c.axis(1).pos == 200)
        assert c.stats()['steps_total'] == 200
    finally:
        c.stop()
    assert not c.alive


# ── retargeting mid-move (click-to-aim while moving) ─────────────────────────
def _max_dv(ax: AxisKinematics, seconds: float, dt: float = DT) -> float:
    """Largest velocity change in one tick, ignoring the final on-the-spot stop."""
    worst, v = 0.0, ax.v
    for _ in range(int(seconds / dt)):
        ax.update(dt)
        if ax.moving:
            worst = max(worst, abs(ax.v - v))
        v = ax.v
    return worst


def test_retarget_behind_decelerates_and_reverses():
    ax = AxisKinematics(accel=2000)
    ax.goto(10000, 1000)
    run(ax, 1.0)                          # cruising forward at 1000
    ax.goto(ax.pos - 200, 1000)           # new target behind the axis
    assert _max_dv(ax, 4.0) <= 2000 * DT + 1e-9
    assert ax.pos == ax.target and not ax.moving


def test_retarget_inside_stopping_distance_overshoots_and_returns():
    ax = AxisKinematics(accel=2000)
    ax.goto(10000, 1000)
    run(ax, 1.0)                          # v²/2a = 250 steps to stop
    target = ax.pos + 50
    ax.goto(target, 1000)
    peak = ax.pos
    for _ in range(int(4.0 / DT)):
        ax.update(DT)
        peak = max(peak, ax.pos)
    assert peak > target + 150            # ran past, didn't stop dead
    assert ax.pos == target and not ax.moving


def test_overshoot_never_crosses_a_soft_limit():
    ax = AxisKinematics(accel=2000)
    ax.set_limits(None, 1500)
    ax.goto(1500, 1000)
    run(ax, 1.0)
    ax.goto(ax.pos + 10, 1000)            # inside stopping distance, next to the limit
    for _ in range(int(4.0 / DT)):
        ax.update(DT)
        assert ax.pos <= 1500


def test_limits_set_while_outside_do_not_jump():
    ax = AxisKinematics(accel=2000)
    ax.set_position(500)
    ax.set_limits(0, 300)
    ax.velocity(1000)                     # drives back to the limit, step by step
    steps = run(ax, 2.0)
    assert ax.pos == 300
    assert max(abs(s) for s in steps) <= 1
