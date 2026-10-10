"""
Multi-axis step generation for turretd.

The pan, tilt and loader steppers are driven from ONE loop that runs in its own
process (pinned to a CPU core when possible), so camera, network and Python's
GIL in the main process can't disturb the step timing.

  * AxisKinematics — pure motion model of one axis (no I/O, easy to test).
    Velocity mode ("hold to move") and position mode (go to a target) with a
    linear acceleration ramp; soft limits turn a velocity move into a position
    move toward the limit, so the axis decelerates and stops exactly there.
    Step generation is a phase accumulator: phase += v·dt, a step is emitted
    every time it crosses ±1.

  * MotionWorker — the loop: drains commands, advances every axis, pulses
    STEP/DIR/EN through a pin backend, publishes status to shared memory.

  * MotionClient — what the main process uses: send commands, read status.
    Runs the worker in a separate process (default) or a thread (tests, sim).

All positions and speeds here are in raw µsteps; unit conversion (degrees,
gear ratio, inversion) happens in the controller.
"""

import math
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Status layout per axis in the shared array
_S_POS, _S_VEL, _S_TARGET, _S_ENABLED, _S_MOVING, _S_LO, _S_HI = range(7)
_S_FIELDS = 7
# Global stats after the per-axis block
_G_LOOP_MAX_GAP_US, _G_STEPS_TOTAL, _G_HEARTBEAT = range(3)
_G_FIELDS = 3

_STEP_PULSE_S = 3e-6
_DIR_SETUP_S = 5e-6
_ENABLE_SETTLE_S = 0.002
_CMD_POLL_S = 0.001          # how often a busy loop drains the command queue
_STATUS_EVERY_S = 0.005
_IDLE_WAIT_S = 0.02          # queue wait when nothing moves
_SLEEP_IF_NEXT_STEP_AFTER_S = 0.002
_NO_LIMIT = float('nan')
_STOP_SLACK = 4               # µsteps: a goto may end abruptly from v² ≤ 2·a·this


def _busy_wait(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# ── pure kinematics ──────────────────────────────────────────────────────────
class AxisKinematics:
    """One axis: position (µsteps), signed velocity (µsteps/s), ramped moves."""

    def __init__(self, accel: float = 1000.0, vmax: float = 1000.0) -> None:
        self.pos = 0
        self.v = 0.0
        self.phase = 0.0
        self.mode = 'idle'           # idle | vel | pos
        self.v_cmd = 0.0             # velocity-mode target
        self.target = 0              # position-mode target
        self.vmax = float(vmax)      # position-mode cruise speed
        self.accel = float(accel)
        self.lo: Optional[int] = None
        self.hi: Optional[int] = None
        self.free = False            # current goto ignores the soft limits

    # ── commands ─────────────────────────────────────────────────────────────
    def set_limits(self, lo: Optional[int], hi: Optional[int]) -> None:
        self.lo, self.hi = lo, hi
        if self.mode == 'pos' and not self.free:
            self.target = self._clamp(self.target)

    def _clamp(self, p: int) -> int:
        if self.lo is not None:
            p = max(self.lo, p)
        if self.hi is not None:
            p = min(self.hi, p)
        return p

    def velocity(self, v: float) -> None:
        """Run at signed speed v (µsteps/s); 0 = ramp down and stop."""
        if v == 0:
            if self.mode == 'pos':
                # Stop as soon as possible: target = where the decel ramp ends.
                d = self.v * self.v / (2 * self.accel) if self.accel else 0
                self.goto(self.pos + int(math.copysign(math.ceil(d), self.v)), max(abs(self.v), 1))
                return
            self.v_cmd = 0.0
            if self.mode == 'idle':
                return
            self.mode = 'vel'
            return
        limit = self.hi if v > 0 else self.lo
        if limit is not None:
            self.goto(limit, abs(v))         # stops exactly at the limit
        else:
            self.mode, self.v_cmd = 'vel', float(v)

    def goto(self, target: int, vmax: float, respect_limits: bool = True) -> None:
        """Ramp to target. Safe to call mid-move: if the new target is behind the
        axis or closer than its stopping distance, it decelerates past the target
        and comes back instead of stopping dead."""
        self.free = not respect_limits
        self.target = self._clamp(int(target)) if respect_limits else int(target)
        self.vmax = max(1.0, float(vmax))
        self.mode = 'pos'

    def _can_stop(self) -> bool:
        """Slow enough to stop within _STOP_SLACK µsteps, i.e. stop on the spot."""
        return self.v * self.v <= 2 * self.accel * _STOP_SLACK

    def halt(self) -> None:
        """Stop instantly (no ramp): emergency stop."""
        self.mode, self.v, self.v_cmd, self.phase = 'idle', 0.0, 0.0, 0.0

    def set_position(self, pos: int) -> None:
        self.halt()
        self.pos = int(pos)
        self.target = self.pos

    @property
    def moving(self) -> bool:
        return self.mode != 'idle'

    # ── simulation step ──────────────────────────────────────────────────────
    def _desired_velocity(self) -> float:
        if self.mode == 'vel':
            return self.v_cmd
        if self.mode == 'pos':
            d = self.target - self.pos
            if d == 0:
                return 0.0
            return math.copysign(min(self.vmax, math.sqrt(2 * self.accel * abs(d))), d)
        return 0.0

    def update(self, dt: float) -> int:
        """Advance by dt seconds. Returns the signed number of steps to emit."""
        if self.mode == 'idle':
            return 0
        v_des = self._desired_velocity()
        dv = self.accel * dt
        self.v += max(-dv, min(dv, v_des - self.v))

        self.phase += self.v * dt
        steps = int(self.phase)              # truncates toward zero
        if self.mode == 'pos' and steps:
            d = self.target - self.pos
            # Land on the target if we can stop there; too fast (target moved
            # inside the stopping distance) → run past it and come back.
            if steps * d > 0 and abs(steps) >= abs(d) and self._can_stop():
                steps = d
        if steps and not (self.mode == 'pos' and self.free):
            # Block motion beyond a soft limit (never jump back if already past it).
            clamped = steps
            if steps > 0 and self.hi is not None:
                clamped = min(steps, max(0, self.hi - self.pos))
            elif steps < 0 and self.lo is not None:
                clamped = max(steps, min(0, self.lo - self.pos))
            if clamped != steps:             # overshoot hit a soft limit: stop there
                steps, self.v, self.phase = clamped, 0.0, 0.0
        self.phase -= steps
        self.pos += steps

        # The sqrt ramp brings v near zero at the target, so stopping there is
        # smooth; for tiny moves it simply ends on arrival.
        if self.mode == 'pos' and self.pos == self.target and self._can_stop():
            self.halt()
        elif self.mode == 'vel' and self.v_cmd == 0 and abs(self.v) <= dv:
            self.halt()
        return steps

    def time_to_next_step(self) -> float:
        if self.mode == 'idle' or self.v == 0:
            return math.inf
        remaining = 1 - abs(self.phase)
        return remaining / abs(self.v)


# ── pin backends ─────────────────────────────────────────────────────────────
class SimPins:
    """No hardware: remembers levels and counts STEP pulses."""

    def __init__(self) -> None:
        self.level: dict[int, int] = {}
        self.pulses: dict[int, int] = {}

    def setup_output(self, pin: int, initial: int) -> None:
        self.level[pin] = initial

    def write(self, pin: int, value: int) -> None:
        if value and not self.level.get(pin):
            self.pulses[pin] = self.pulses.get(pin, 0) + 1
        self.level[pin] = value

    def close(self, en_pins=()) -> None:
        pass


class GpioPins:
    """RPi.GPIO (rpi-lgpio on trixie), BCM numbering. Imported in the worker."""

    def __init__(self) -> None:
        import RPi.GPIO as GPIO
        self.GPIO = GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

    def setup_output(self, pin: int, initial: int) -> None:
        self.GPIO.setup(pin, self.GPIO.OUT, initial=initial)

    def write(self, pin: int, value: int) -> None:
        self.GPIO.output(pin, value)

    def close(self, en_pins=()) -> None:
        self.GPIO.cleanup()
        # Leave EN pulled up: a floating EN reads LOW through the DRV8825's
        # internal pull-down, which would enable (energise) the drivers.
        if en_pins:
            try:
                subprocess.run(['pinctrl', 'set', ','.join(map(str, en_pins)), 'ip', 'pu'],
                               capture_output=True, timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass


@dataclass(frozen=True)
class AxisPins:
    step: int
    dir: int
    en: int            # active LOW


# ── worker loop ──────────────────────────────────────────────────────────────
class MotionWorker:
    def __init__(self, pins_cfg: list[AxisPins], pin_backend, cmd_q, status, *,
                 realtime: bool = False) -> None:
        self.pins_cfg = pins_cfg
        self.pins = pin_backend
        self.cmd_q = cmd_q
        self.status = status
        self.realtime = realtime
        n = len(pins_cfg)
        self.axes = [AxisKinematics() for _ in range(n)]
        self.enabled = [False] * n
        self.dir_level = [0] * n
        self.hold = [True] * n
        self.idle_disable_s = [30.0] * n
        self.idle_since = [time.monotonic()] * n
        self.running = True
        self.orphaned = False
        self.steps_total = 0
        self.max_gap = 0.0
        self._ppid = os.getppid()

    # ── setup ────────────────────────────────────────────────────────────────
    def _setup(self) -> None:
        if self.realtime:
            try:
                os.sched_setaffinity(0, {os.cpu_count() - 1})
            except (AttributeError, OSError):
                pass
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
            except (AttributeError, OSError, PermissionError):
                try:
                    os.nice(-10)
                except OSError:
                    pass
        for p in self.pins_cfg:
            self.pins.setup_output(p.en, 1)        # disabled
            self.pins.setup_output(p.step, 0)
            self.pins.setup_output(p.dir, 0)

    # ── commands ─────────────────────────────────────────────────────────────
    def _enable(self, i: int, on: bool) -> None:
        if on == self.enabled[i]:
            return
        self.pins.write(self.pins_cfg[i].en, 0 if on else 1)
        self.enabled[i] = on
        if on:
            _busy_wait(_ENABLE_SETTLE_S)
        else:
            self.axes[i].halt()

    def handle(self, cmd: tuple) -> None:
        op = cmd[0]
        if op == 'halt_all':
            for ax in self.axes:
                ax.halt()
            return
        if op == 'shutdown':
            self.running = False
            return
        i = cmd[1]
        ax = self.axes[i]
        if op == 'vel':
            if cmd[2] != 0:
                self._enable(i, True)
            ax.velocity(cmd[2])
        elif op == 'goto':
            self._enable(i, True)
            ax.goto(cmd[2], cmd[3])
        elif op == 'move_by':
            self._enable(i, True)
            ax.goto(ax.pos + int(cmd[2]), cmd[3])
        elif op == 'raw_move_by':           # diagnostics: ignores soft limits
            self._enable(i, True)
            ax.goto(ax.pos + int(cmd[2]), cmd[3], respect_limits=False)
        elif op == 'stop':
            ax.velocity(0)
        elif op == 'halt':
            ax.halt()
        elif op == 'set_position':
            ax.set_position(cmd[2])
        elif op == 'limits':
            ax.set_limits(cmd[2], cmd[3])
        elif op == 'params':
            p = cmd[2]
            if 'accel' in p:
                ax.accel = max(1.0, float(p['accel']))
            if 'hold' in p:
                self.hold[i] = bool(p['hold'])
            if 'idle_disable' in p:
                self.idle_disable_s[i] = float(p['idle_disable'])
        elif op == 'enable':
            self._enable(i, bool(cmd[2]))
        self.idle_since[i] = time.monotonic()

    def _drain(self, block: bool) -> None:
        try:
            cmd = self.cmd_q.get(timeout=_IDLE_WAIT_S) if block else self.cmd_q.get_nowait()
        except queue.Empty:
            return
        self.handle(cmd)
        while True:
            try:
                self.handle(self.cmd_q.get_nowait())
            except queue.Empty:
                return

    # ── status ───────────────────────────────────────────────────────────────
    def _publish(self) -> None:
        s = self.status
        for i, ax in enumerate(self.axes):
            b = i * _S_FIELDS
            s[b + _S_POS] = ax.pos
            s[b + _S_VEL] = ax.v
            s[b + _S_TARGET] = ax.target if ax.mode == 'pos' else ax.pos
            s[b + _S_ENABLED] = 1.0 if self.enabled[i] else 0.0
            s[b + _S_MOVING] = 1.0 if ax.moving else 0.0
            s[b + _S_LO] = _NO_LIMIT if ax.lo is None else ax.lo
            s[b + _S_HI] = _NO_LIMIT if ax.hi is None else ax.hi
        g = len(self.axes) * _S_FIELDS
        s[g + _G_LOOP_MAX_GAP_US] = self.max_gap * 1e6
        s[g + _G_STEPS_TOTAL] = self.steps_total
        s[g + _G_HEARTBEAT] = time.monotonic()

    def _idle_disable(self, now: float) -> None:
        for i, ax in enumerate(self.axes):
            if ax.moving:
                self.idle_since[i] = now
            elif (self.enabled[i] and not self.hold[i]
                  and now - self.idle_since[i] > self.idle_disable_s[i]):
                self._enable(i, False)

    def _emit(self, i: int, steps: int) -> None:
        p = self.pins_cfg[i]
        level = 1 if steps > 0 else 0
        if level != self.dir_level[i]:
            self.pins.write(p.dir, level)
            self.dir_level[i] = level
            _busy_wait(_DIR_SETUP_S)
        for k in range(abs(steps)):
            if k:
                _busy_wait(_STEP_PULSE_S)       # DRV8825: STEP low ≥ 1.9 µs too
            self.pins.write(p.step, 1)
            _busy_wait(_STEP_PULSE_S)
            self.pins.write(p.step, 0)
        self.steps_total += abs(steps)

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> None:
        self._setup()
        last = time.perf_counter()
        last_cmd = last_status = last
        gap_window_start = last
        self._publish()
        try:
            while self.running:
                moving = any(ax.moving for ax in self.axes)
                if not moving:
                    if os.getppid() != self._ppid:     # main process died
                        self.orphaned = True
                        break
                    self._drain(block=True)
                    now = time.perf_counter()
                    self._idle_disable(time.monotonic())
                    self._publish()
                    last = last_status = now
                    continue

                now = time.perf_counter()
                dt = now - last
                last = now
                if dt > self.max_gap:
                    self.max_gap = dt
                if now - gap_window_start > 1.0:
                    gap_window_start, self.max_gap = now, dt

                for i, ax in enumerate(self.axes):
                    steps = ax.update(dt)
                    if steps:
                        self._emit(i, steps)

                if now - last_cmd > _CMD_POLL_S:
                    self._drain(block=False)
                    last_cmd = now
                if now - last_status > _STATUS_EVERY_S:
                    self._publish()
                    last_status = now
                    if os.getppid() != self._ppid:
                        self.orphaned = True
                        break

                # Long until the next step (slow moves): give the CPU a break.
                nxt = min(ax.time_to_next_step() for ax in self.axes)
                if nxt > _SLEEP_IF_NEXT_STEP_AFTER_S:
                    time.sleep(min(nxt / 2, 0.001))
        finally:
            for i in range(len(self.axes)):
                self.pins.write(self.pins_cfg[i].step, 0)
                self.pins.write(self.pins_cfg[i].en, 1)
                self.enabled[i] = False
            self._publish()
            self.pins.close([p.en for p in self.pins_cfg])


def _worker_main(pins_cfg, backend: str, cmd_q, status, realtime: bool, on_orphan) -> None:
    pins = GpioPins() if backend == 'gpio' else SimPins()
    worker = MotionWorker(pins_cfg, pins, cmd_q, status, realtime=realtime)
    worker.run()
    if worker.orphaned and on_orphan is not None:
        on_orphan()


# ── client (main process) ────────────────────────────────────────────────────
@dataclass
class AxisStatus:
    pos: int
    vel: float
    target: int
    enabled: bool
    moving: bool
    lo: Optional[int]
    hi: Optional[int]


class MotionClient:
    """Start the worker and talk to it. backend: 'gpio' or 'sim'."""

    def __init__(self, pins_cfg: list[AxisPins], backend: str = 'sim', *,
                 use_process: bool = True, on_orphan=None) -> None:
        """:param on_orphan: called in the worker process if the main process dies
                          (e.g. to switch off hardware PWM). Module-level function."""
        self.n = len(pins_cfg)
        self.backend = backend
        size = self.n * _S_FIELDS + _G_FIELDS
        if use_process:
            ctx = mp.get_context('fork' if hasattr(os, 'fork') else 'spawn')
            self.cmd_q = ctx.Queue()
            self.status = ctx.Array('d', size, lock=False)
            self._runner = ctx.Process(target=_worker_main, daemon=True,
                                       args=(pins_cfg, backend, self.cmd_q, self.status,
                                             backend == 'gpio', on_orphan))
        else:
            self.cmd_q = queue.Queue()
            self.status = [0.0] * size
            pins = GpioPins() if backend == 'gpio' else SimPins()
            self.worker = MotionWorker(pins_cfg, pins, self.cmd_q, self.status)
            self._runner = threading.Thread(target=self.worker.run, daemon=True)

    def start(self) -> None:
        self._runner.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.cmd_q.put(('shutdown',))
        self._runner.join(timeout)
        if isinstance(self._runner, mp.process.BaseProcess) and self._runner.is_alive():
            self._runner.terminate()

    @property
    def alive(self) -> bool:
        return self._runner.is_alive()

    def send(self, *cmd) -> None:
        self.cmd_q.put(tuple(cmd))

    def axis(self, i: int) -> AxisStatus:
        b = i * _S_FIELDS
        s = self.status
        lo, hi = s[b + _S_LO], s[b + _S_HI]
        return AxisStatus(
            pos=int(s[b + _S_POS]), vel=s[b + _S_VEL], target=int(s[b + _S_TARGET]),
            enabled=s[b + _S_ENABLED] > 0.5, moving=s[b + _S_MOVING] > 0.5,
            lo=None if lo != lo else int(lo), hi=None if hi != hi else int(hi),
        )

    def stats(self) -> dict:
        g = self.n * _S_FIELDS
        return {
            'loop_max_gap_us': round(self.status[g + _G_LOOP_MAX_GAP_US], 1),
            'steps_total': int(self.status[g + _G_STEPS_TOTAL]),
            'worker_heartbeat': self.status[g + _G_HEARTBEAT],
        }
