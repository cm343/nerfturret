"""
Turret controller: state machine, safety, units, fire cycle, telemetry.

States
  DISARMED  pan/tilt may move; flywheels and loader locked out (default).
  ARMED     flywheels may spin; firing allowed once they are up to speed.
  E-STOP    latched on top of DISARMED (estop_reason set): everything halted,
            nothing moves until reset().

Safety
  * Watchdog: while anything is active (armed, spinning, firing, an axis
    moving), a heartbeat from the UI must arrive every safety.watchdog_ms,
    otherwise → E-STOP "link lost".
  * Soft limits on pan/tilt once the axis is referenced (zero set by hand —
    there are no endstops). Unreferenced axes: no goto, no fast moves.
  * Fire only when ARMED, loader referenced and flywheels spun up for
    flywheel.spinup_time. Auto mode stops after fire.max_fire_time.

Units: the UI speaks degrees at the output (after gear ratio and inversion);
the motion worker speaks µsteps. The loader works in motor degrees.
"""

import asyncio
import math
import time
from typing import Callable, Optional

from . import system
from .aim import Aimer, AimAbort
from .config import PARAMS, ConfigStore
from .motion import AxisPins, MotionClient

AXES = ('pan', 'tilt', 'loader')
AXIS_INDEX = {a: i for i, a in enumerate(AXES)}
AIM_AXES = ('pan', 'tilt')

# BCM pins, nerf-turret-wiring.html Rev C
AXIS_PINS = {
    'pan':    AxisPins(step=16, dir=20, en=6),
    'tilt':   AxisPins(step=23, dir=24, en=5),
    'loader': AxisPins(step=17, dir=27, en=4),
}
FULL_STEPS_PER_REV = 200
TICK_S = 0.02                 # 50 Hz control tick (= one ESC frame)
SPINUP_TIMEOUT_S = 5.0        # extra time allowed for the wheels to reach speed


def frame_angle(frac: float, fov_deg: float) -> float:
    """Angle (°) off the optical axis of a point at `frac` of the half-frame,
    for a pinhole camera: tan(angle) = frac · tan(fov/2), not frac · fov/2."""
    return math.degrees(math.atan(frac * math.tan(math.radians(fov_deg) / 2)))


class CommandError(Exception):
    pass


class _Abort(Exception):
    pass


class Controller:
    def __init__(self, cfg: ConfigStore, motion: MotionClient, flywheels, camera=None, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.motion = motion
        self.fly = flywheels
        self.camera = camera
        self.clock = clock
        self.aim = Aimer(self)

        self.state = 'DISARMED'
        self.estop_reason: Optional[str] = None
        self.arming = False
        self.referenced = {'pan': False, 'tilt': False,
                           'loader': bool(cfg['loader.assume_rest_on_start'])}
        self.spin_on = False
        self.spin_ready_since: Optional[float] = None
        self.last_activity = clock()
        self.fire_task: Optional[asyncio.Task] = None
        self.fire_stop_requested = False
        self.shots = 0
        self.ammo = cfg['loader.magazine']
        self.last_heartbeat: Optional[float] = None
        self.restart_needed: set[str] = set()
        self.listeners: list[Callable[[dict], None]] = []
        self._tasks: list[asyncio.Task] = []
        self._arm_task: Optional[asyncio.Task] = None

    # ── events ───────────────────────────────────────────────────────────────
    def emit(self, level: str, msg: str) -> None:
        ev = {'type': 'event', 'level': level, 'msg': msg, 't': time.time()}
        for fn in list(self.listeners):
            fn(ev)

    def broadcast(self, message: dict) -> None:
        for fn in list(self.listeners):
            fn(message)

    # ── units ────────────────────────────────────────────────────────────────
    def steps_per_deg(self, axis: str) -> float:
        micro = int(self.cfg[f'{axis}.microsteps'])
        gear = float(self.cfg.get(f'{axis}.gear_ratio', 1.0))
        sign = -1.0 if self.cfg[f'{axis}.invert'] else 1.0
        return sign * FULL_STEPS_PER_REV * micro * gear / 360.0

    def to_steps(self, axis: str, deg: float) -> int:
        return int(round(deg * self.steps_per_deg(axis)))

    def to_deg(self, axis: str, steps: float) -> float:
        return steps / self.steps_per_deg(axis)

    def speed_steps(self, axis: str, deg_per_s: float) -> float:
        return abs(deg_per_s * self.steps_per_deg(axis))

    def position_deg(self, axis: str) -> float:
        return self.to_deg(axis, self.motion.axis(AXIS_INDEX[axis]).pos)

    @staticmethod
    def axis_index(axis: str) -> int:
        return AXIS_INDEX[axis]

    # ── aiming helpers (used by click-to-aim and aim.py) ─────────────────────
    def view_delta(self, u: float, v: float, cx: float, cy: float) -> tuple[float, float]:
        """Pan/tilt turn (°) that brings image point (u, v) onto (cx, cy)."""
        hfov, vfov = self.cfg['camera.hfov'], self.cfg['camera.vfov']
        return (frame_angle(u, hfov) - frame_angle(cx, hfov),
                -(frame_angle(v, vfov) - frame_angle(cy, vfov)))

    def move_by_deg(self, d_pan: float = 0.0, d_tilt: float = 0.0) -> None:
        for axis, deg in (('pan', d_pan), ('tilt', d_tilt)):
            if abs(deg) > 0.01:
                self.motion.send('move_by', AXIS_INDEX[axis], self.to_steps(axis, deg),
                                 self.speed_steps(axis, self.cfg[f'{axis}.speed']))

    def move_open_loop(self, u: float, v: float, cx: float, cy: float) -> None:
        self.move_by_deg(*self.view_delta(u, v, cx, cy))

    def goto_deg(self, axis: str, deg: float) -> None:
        self.motion.send('goto', AXIS_INDEX[axis], self.to_steps(axis, deg),
                         self.speed_steps(axis, self.cfg[f'{axis}.speed']))

    # ── configuration → hardware ─────────────────────────────────────────────
    def apply_axis(self, axis: str) -> None:
        i = AXIS_INDEX[axis]
        self.motion.send('params', i, {
            'accel': self.speed_steps(axis, self.cfg[f'{axis}.accel']),
            'hold': self.cfg[f'{axis}.hold_when_idle'],
            'idle_disable': self.cfg[f'{axis}.idle_disable'],
        })
        self.apply_limits(axis)

    def limits_steps(self, axis: str) -> tuple[Optional[int], Optional[int]]:
        if (axis not in AIM_AXES or not self.referenced[axis]
                or not self.cfg[f'{axis}.limits_enabled']):
            return None, None
        a = self.to_steps(axis, self.cfg[f'{axis}.limit_min'])
        b = self.to_steps(axis, self.cfg[f'{axis}.limit_max'])
        return min(a, b), max(a, b)

    def apply_limits(self, axis: str) -> None:
        lo, hi = self.limits_steps(axis)
        self.motion.send('limits', AXIS_INDEX[axis], lo, hi)

    def set_config(self, changes: dict) -> dict:
        applied = self.cfg.set_many(changes)
        self._after_config_change(applied)
        self.broadcast({'type': 'config', 'values': applied})
        return applied

    def load_preset(self, name: str) -> dict:
        changed = self.cfg.load_preset(name)
        self._after_config_change(changed)
        self.broadcast({'type': 'config', 'values': changed})
        self.emit('info', f'preset "{name}" loaded ({len(changed)} changes)')
        return changed

    def _after_config_change(self, changed: dict) -> None:
        axes, camera = set(), False
        for key in changed:
            p = PARAMS[key]
            if p.apply == 'restart':
                self.restart_needed.add(key)
            elif p.apply == 'camera':
                camera = True
            if p.group in AXES:
                axes.add(p.group)
        for axis in axes:
            self.apply_axis(axis)
        if 'loader.magazine' in changed and changed['loader.magazine'] > 0:
            self.ammo = min(self.ammo, changed['loader.magazine'])
        if camera and self.camera is not None:
            asyncio.get_running_loop().run_in_executor(
                None, self.camera.restart, self.cfg.snapshot())
        if self.restart_needed & set(changed):
            self.emit('warn', 'restart turretd to apply: '
                      + ', '.join(sorted(self.restart_needed & set(changed))))

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self) -> None:
        for axis in AXES:
            self.apply_axis(axis)
        self.fly.setup()
        self._tasks.append(asyncio.create_task(self._tick_loop()))

    async def shutdown(self) -> None:
        self.estop('turretd shutting down', quiet=True)
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    # ── safety ───────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return (self.state == 'ARMED' or self.arming or self.spin_on or self.firing
                or any(self.motion.axis(i).moving for i in range(len(AXES))))

    @property
    def firing(self) -> bool:
        return self.fire_task is not None and not self.fire_task.done()

    def estop(self, reason: str, quiet: bool = False) -> None:
        self.motion.send('halt_all')
        self.aim.cancel()
        self.fly.stop()
        self.spin_on = False
        self.spin_ready_since = None
        self.state = 'DISARMED'
        self.arming = False
        for t in (self.fire_task, self._arm_task):
            if t is not None and not t.done():
                t.cancel()
        if self.estop_reason is None:
            self.estop_reason = reason
            if not quiet:
                self.emit('error', f'E-STOP: {reason}')

    def reset(self) -> None:
        if self.estop_reason is None:
            return
        self.estop_reason = None
        self.emit('info', 'E-STOP reset')
        self._return_loader()

    def disarm(self, reason: str = 'disarmed') -> None:
        was = self.state == 'ARMED' or self.arming
        self.spin_on = False
        self.spin_ready_since = None
        self.fly.stop()
        if self.fire_task is not None and not self.fire_task.done():
            self.fire_task.cancel()
        if self._arm_task is not None and not self._arm_task.done():
            self._arm_task.cancel()
        self.arming = False
        self.state = 'DISARMED'
        if was:
            self.emit('info', reason)

    def _return_loader(self) -> None:
        st = self.motion.axis(AXIS_INDEX['loader'])
        if self.referenced['loader'] and st.pos != 0 and self.estop_reason is None:
            self.motion.send('goto', AXIS_INDEX['loader'], 0,
                             self.speed_steps('loader', self.cfg['loader.return_speed']))
            self.emit('info', 'loader returning to rest')

    def _check_watchdog(self, now: float) -> None:
        if self.estop_reason is not None or not self.active:
            return
        timeout = self.cfg['safety.watchdog_ms'] / 1000
        if self.last_heartbeat is None or now - self.last_heartbeat > timeout:
            self.estop(f'link lost (no heartbeat for {self.cfg["safety.watchdog_ms"]} ms)')

    # ── flywheels ────────────────────────────────────────────────────────────
    def wheel_targets(self) -> tuple[float, float]:
        if not (self.state == 'ARMED' and self.spin_on):
            return 0.0, 0.0
        thr, trim = self.cfg['flywheel.throttle'], self.cfg['flywheel.trim']
        cap = self.cfg['flywheel.max_throttle']
        clamp = lambda x: max(0.0, min(cap, x)) / 100.0   # noqa: E731
        return clamp(thr + trim / 2), clamp(thr - trim / 2)

    @property
    def spun_up(self) -> bool:
        return (self.spin_on and self.spin_ready_since is not None
                and self.clock() - self.spin_ready_since >= self.cfg['flywheel.spinup_time'])

    def _flywheel_tick(self, now: float) -> None:
        targets = self.wheel_targets()
        if targets == (0.0, 0.0):
            if any(t > 0 for t in self.fly.throttles):
                self.fly.stop()
            self.spin_ready_since = None
            return
        rate = self.cfg['flywheel.ramp_rate'] / 100.0
        if self.fly.ramp_tick(targets, rate * TICK_S):
            if self.spin_ready_since is None:
                self.spin_ready_since = now
        else:
            self.spin_ready_since = None
        idle = self.cfg['flywheel.idle_spindown']
        if idle > 0 and not self.firing and now - self.last_activity > idle:
            self.spin_on = False
            self.emit('info', f'flywheels spun down after {idle:.0f} s idle')

    async def _tick_loop(self) -> None:
        while True:
            now = self.clock()
            if not self.motion.alive and self.estop_reason is None:
                self.estop('motion worker died — check the turretd log, then restart turretd')
            self._check_watchdog(now)
            self._flywheel_tick(now)
            self.aim.tick()
            await asyncio.sleep(TICK_S)

    # ── fire cycle ───────────────────────────────────────────────────────────
    def _check_can_continue(self, require_armed: bool = True) -> None:
        if self.estop_reason is not None or (require_armed and self.state != 'ARMED'):
            raise _Abort()

    async def _wait_loader(self, target: int, timeout: float, require_armed: bool = True) -> None:
        i = AXIS_INDEX['loader']
        end = self.clock() + timeout
        while True:
            self._check_can_continue(require_armed)
            st = self.motion.axis(i)
            if st.pos == target and not st.moving:
                return
            if self.clock() > end:
                self.estop('loader did not reach its position in time')
                raise _Abort()
            await asyncio.sleep(0.005)

    def _move_time(self, deg: float, speed: float, accel: float) -> float:
        return abs(deg) / speed + speed / accel

    async def _loader_cycle(self, speed_scale: float = 1.0, require_armed: bool = True) -> None:
        c, i = self.cfg, AXIS_INDEX['loader']
        stroke = c['loader.stroke']
        front = self.to_steps('loader', stroke)
        push_deg = c['loader.push_speed'] * speed_scale
        back_deg = c['loader.return_speed'] * speed_scale
        self.motion.send('goto', i, front, self.speed_steps('loader', push_deg))
        await self._wait_loader(front, 2 + 3 * self._move_time(stroke, push_deg, c['loader.accel']),
                                require_armed)
        await asyncio.sleep(c['loader.front_dwell'] / 1000)
        self._check_can_continue(require_armed)
        self.motion.send('goto', i, 0, self.speed_steps('loader', back_deg))
        await self._wait_loader(0, 2 + 3 * self._move_time(stroke, back_deg, c['loader.accel']),
                                require_armed)

    async def _wait_spun_up(self) -> None:
        end = self.clock() + self.cfg['flywheel.spinup_time'] + SPINUP_TIMEOUT_S
        while not self.spun_up:
            self._check_can_continue()
            if self.clock() > end:
                self.emit('warn', 'flywheels did not reach speed — not firing')
                raise _Abort()
            await asyncio.sleep(0.01)

    async def _fire(self, mode: str, wheels: bool) -> None:
        """Run loader cycles. wheels=False: load only (flywheels not used), which
        also works while DISARMED; wheels=True: wait for spin-up before each push."""
        c = self.cfg
        count = {'single': 1, 'burst': c['loader.burst'], 'auto': math.inf}[mode]
        t0 = self.clock()
        fired = 0
        verb = 'fired' if wheels else 'loaded'
        try:
            while fired < count:
                if mode == 'auto' and self.fire_stop_requested:
                    break
                if mode == 'auto' and self.clock() - t0 > c['fire.max_fire_time']:
                    self.emit('warn', f'max continuous fire time ({c["fire.max_fire_time"]:.0f} s) reached')
                    break
                if c['loader.magazine'] > 0 and self.ammo <= 0:
                    self.emit('warn', 'magazine empty — reload')
                    break
                if wheels:
                    await self._wait_spun_up()
                start = self.clock()
                await self._loader_cycle(require_armed=wheels)
                fired += 1
                self.shots += 1
                if c['loader.magazine'] > 0:
                    self.ammo -= 1
                self.last_activity = self.clock()
                if fired < count:
                    wait = 1.0 / c['loader.rate'] - (self.clock() - start)
                    if wait > 0:
                        await asyncio.sleep(wait)
        except (_Abort, asyncio.CancelledError):
            if self.estop_reason is None:
                self._return_loader()
            if fired:
                self.emit('info', f'{verb} {fired}')
            return
        finally:
            self.fire_stop_requested = False
        self.emit('info', f'{verb} {fired}' + ('' if wheels else ' (flywheels off)'))

    # ── commands ─────────────────────────────────────────────────────────────
    def _require_no_estop(self) -> None:
        if self.estop_reason is not None:
            raise CommandError(f'E-STOP active ({self.estop_reason}) — reset first')

    def _axis(self, cmd: dict, allowed=AXES) -> str:
        axis = cmd.get('axis')
        if axis not in allowed:
            raise CommandError(f'axis must be one of {list(allowed)}')
        return axis

    async def handle(self, cmd: dict) -> Optional[dict]:
        """Execute one UI command. Raises CommandError with a user-facing message."""
        t = cmd.get('type')
        # Any command proves the link is alive, not just explicit heartbeats.
        self.last_heartbeat = self.clock()
        fn = getattr(self, f'_cmd_{t}', None) if isinstance(t, str) else None
        if fn is None:
            raise CommandError(f'unknown command {t!r}')
        return await fn(cmd)

    async def _cmd_heartbeat(self, cmd) -> None:
        self.last_heartbeat = self.clock()

    async def _cmd_estop(self, cmd) -> None:
        self.estop(cmd.get('reason') or 'E-STOP pressed')

    async def _cmd_reset(self, cmd) -> None:
        if not self.motion.alive:
            raise CommandError('motion worker is not running — restart turretd')
        self.reset()

    async def _cmd_arm(self, cmd) -> None:
        self._require_no_estop()
        if self.state == 'ARMED' or self.arming:
            return
        self.arming = True
        self.emit('info', 'arming…' if not self.fly.armed else 'armed')

        async def arm():
            try:
                await self.fly.arm(self.cfg['flywheel.arm_time'])
            except asyncio.CancelledError:
                return
            if self.arming and self.estop_reason is None:
                self.arming = False
                self.state = 'ARMED'
                self.last_activity = self.clock()
                self.emit('info', 'ARMED')

        self._arm_task = asyncio.create_task(arm())

    async def _cmd_disarm(self, cmd) -> None:
        self.disarm()

    async def _cmd_spin(self, cmd) -> None:
        on = bool(cmd.get('on'))
        if on:
            self._require_no_estop()
            if self.state != 'ARMED':
                raise CommandError('arm first')
            self.last_activity = self.clock()
        self.spin_on = on

    def fire_uses_wheels(self) -> Optional[bool]:
        """What FIRE would do now: True = shoot (flywheels), False = load only,
        None = not allowed in this state."""
        if self.state == 'ARMED':
            return self.spin_on or bool(self.cfg['fire.auto_spinup'])
        if self.state == 'DISARMED' and not self.arming and self.cfg['fire.load_when_disarmed']:
            return False
        return None

    async def _cmd_fire_start(self, cmd) -> None:
        self._require_no_estop()
        wheels = self.fire_uses_wheels()
        if wheels is None:
            raise CommandError('arm first')
        if not self.referenced['loader']:
            raise CommandError('loader position unknown — set its rest position (Calibration)')
        if self.firing:
            return
        mode = cmd.get('mode') or self.cfg['fire.mode']
        if mode not in ('single', 'burst', 'auto'):
            raise CommandError('mode must be single, burst or auto')
        if wheels and not self.spin_on:
            self.spin_on = True                   # fire.auto_spinup
        self.last_activity = self.clock()
        self.fire_stop_requested = False
        self.fire_task = asyncio.create_task(self._fire(mode, wheels))

    async def _cmd_fire_stop(self, cmd) -> None:
        self.fire_stop_requested = True

    async def _cmd_reload(self, cmd) -> None:
        self.ammo = self.cfg['loader.magazine']

    def _speed_factor(self, axis: str, speed: str) -> float:
        if speed == 'fast' and self.referenced[axis]:
            return self.cfg[f'{axis}.fast_factor']
        if speed == 'fine':
            return self.cfg[f'{axis}.fine_factor']
        return 1.0

    async def _cmd_move(self, cmd) -> None:
        """Hold-to-move: dir −1/0/+1 until the next move command for this axis."""
        axis = self._axis(cmd, AIM_AXES)
        d = cmd.get('dir', 0)
        if d not in (-1, 0, 1):
            raise CommandError('dir must be -1, 0 or 1')
        self.aim.cancel()
        if d == 0:
            self.motion.send('stop', AXIS_INDEX[axis])
            return
        self._require_no_estop()
        deg_s = self.cfg[f'{axis}.speed'] * self._speed_factor(axis, cmd.get('speed', 'normal'))
        v = d * self.speed_steps(axis, deg_s) * (1 if self.steps_per_deg(axis) > 0 else -1)
        self.motion.send('vel', AXIS_INDEX[axis], v)

    async def _cmd_jog(self, cmd) -> None:
        axis = self._axis(cmd, AIM_AXES)
        self._require_no_estop()
        self.aim.cancel()
        d = 1 if cmd.get('dir', 1) > 0 else -1
        deg = float(cmd.get('deg', self.cfg[f'{axis}.jog_step']))
        speed = self.cfg[f'{axis}.speed'] * self._speed_factor(axis, cmd.get('speed', 'normal'))
        self.motion.send('move_by', AXIS_INDEX[axis], self.to_steps(axis, d * deg),
                         self.speed_steps(axis, speed))

    async def _cmd_goto(self, cmd) -> None:
        axis = self._axis(cmd, AIM_AXES)
        self._require_no_estop()
        if not self.referenced[axis]:
            raise CommandError(f'{axis} is not referenced — set its zero first')
        self.aim.cancel()
        try:
            deg = float(cmd['deg'])
        except (KeyError, TypeError, ValueError):
            raise CommandError('deg must be a number') from None
        self.motion.send('goto', AXIS_INDEX[axis], self.to_steps(axis, deg),
                         self.speed_steps(axis, self.cfg[f'{axis}.speed']))

    async def _cmd_aim_offset(self, cmd) -> dict:
        """Turn so the clicked video point lands on the crosshair; with
        aim.mode = auto, then fine-tune by tracking the clicked spot.
        x, y in [-1, 1] relative to the crosshair, +x right, +y down."""
        self._require_no_estop()
        try:
            x, y = float(cmd.get('x', 0)), float(cmd.get('y', 0))
        except (TypeError, ValueError):
            raise CommandError('x and y must be numbers') from None
        return {'mode': await self.aim.click(x, y)}

    async def _cmd_fov_calibrate(self, cmd) -> None:
        """Automatic FOV calibration by tracking the feature at image point x, y."""
        axis = self._axis(cmd, AIM_AXES)
        self._require_no_estop()
        try:
            self.aim.calibrate(axis, float(cmd['x']), float(cmd['y']))
        except (KeyError, TypeError, ValueError):
            raise CommandError('x and y must be numbers') from None
        except AimAbort as e:
            raise CommandError(str(e)) from None

    async def _cmd_aim_cancel(self, cmd) -> None:
        self.aim.cancel()

    async def _cmd_set_zero(self, cmd) -> None:
        axis = self._axis(cmd, AIM_AXES)
        i = AXIS_INDEX[axis]
        if self.motion.axis(i).moving:
            raise CommandError(f'{axis} is moving')
        self.motion.send('enable', i, True)     # hold the reference
        self.motion.send('set_position', i, 0)
        self.referenced[axis] = True
        self.apply_limits(axis)
        self.emit('info', f'{axis} referenced: this position is 0°')

    async def _cmd_set_limit(self, cmd) -> None:
        axis = self._axis(cmd, AIM_AXES)
        which = cmd.get('which')
        if which not in ('min', 'max'):
            raise CommandError('which must be min or max')
        if not self.referenced[axis]:
            raise CommandError(f'reference {axis} first')
        deg = round(self.position_deg(axis), 2)
        other = self.cfg[f'{axis}.limit_{"max" if which == "min" else "min"}']
        if (which == 'min' and deg >= other) or (which == 'max' and deg <= other):
            raise CommandError(f'{which} limit must be {"below" if which == "min" else "above"} {other}°')
        self.set_config({f'{axis}.limit_{which}': deg})
        self.emit('info', f'{axis} {which} limit = {deg}°')

    async def _cmd_enable(self, cmd) -> None:
        axis = self._axis(cmd)
        on = bool(cmd.get('on'))
        if on:
            self._require_no_estop()
        self.motion.send('enable', AXIS_INDEX[axis], on)
        if not on and axis in AIM_AXES and self.referenced[axis]:
            self.referenced[axis] = False
            self.apply_limits(axis)
            self.emit('warn', f'{axis} disabled — reference dropped')

    async def _cmd_raw_jog(self, cmd) -> None:
        """Diagnostics: move by motor degrees, ignoring limits."""
        axis = self._axis(cmd)
        self._require_no_estop()
        if axis == 'loader' and self.firing:
            raise CommandError('loader is firing')
        deg = float(cmd.get('deg', 0))
        micro = int(self.cfg[f'{axis}.microsteps'])
        steps = int(round(deg / 360 * FULL_STEPS_PER_REV * micro))
        speed = FULL_STEPS_PER_REV * micro * 0.25          # 0.25 motor rev/s
        self.motion.send('raw_move_by', AXIS_INDEX[axis], steps, speed)

    async def _cmd_loader_set_rest(self, cmd) -> None:
        i = AXIS_INDEX['loader']
        if self.firing or self.motion.axis(i).moving:
            raise CommandError('loader is moving')
        self.motion.send('set_position', i, 0)
        self.referenced['loader'] = True
        self.emit('info', 'loader rest position set')

    async def _cmd_loader_set_stroke(self, cmd) -> None:
        if not self.referenced['loader']:
            raise CommandError('set the rest position first')
        stroke = abs(round(self.position_deg('loader'), 1))
        if stroke < 10:
            raise CommandError('jog the loader to its front position first')
        self.set_config({'loader.stroke': stroke})
        self.emit('info', f'loader stroke = {stroke}° (motor)')

    async def _cmd_loader_test(self, cmd) -> None:
        """One push-and-return without flywheels (DISARMED only) to check the stroke."""
        if self.state != 'DISARMED' or self.firing:
            raise CommandError('only while disarmed')
        self._require_no_estop()
        if not self.referenced['loader']:
            raise CommandError('set the rest position first')

        async def cycle():
            try:
                await self._loader_cycle(speed_scale=0.25, require_armed=False)
                self.emit('info', 'loader test stroke done')
            except (_Abort, asyncio.CancelledError):
                if self.estop_reason is None:
                    self._return_loader()

        self.fire_task = asyncio.create_task(cycle())

    # ── telemetry ────────────────────────────────────────────────────────────
    def telemetry(self) -> dict:
        now = self.clock()
        axes = {}
        for axis in AXES:
            st = self.motion.axis(AXIS_INDEX[axis])
            spd = self.steps_per_deg(axis)
            lo, hi = self.limits_steps(axis)
            axes[axis] = {
                'pos': round(st.pos / spd, 2), 'target': round(st.target / spd, 2),
                'vel': round(st.vel / spd, 1), 'moving': st.moving, 'enabled': st.enabled,
                'referenced': self.referenced[axis], 'steps': st.pos,
                'limits': None if lo is None else sorted([round(lo / spd, 2), round(hi / spd, 2)]),
            }
        left, right = self.fly.throttles
        return {
            'type': 'telemetry', 't': time.time(),
            'state': 'ESTOP' if self.estop_reason else ('ARMING' if self.arming else self.state),
            'estop_reason': self.estop_reason,
            'axes': axes,
            'flywheels': {
                'on': self.spin_on, 'left': round(left * 100, 1), 'right': round(right * 100, 1),
                'target': [round(x * 100, 1) for x in self.wheel_targets()],
                'spun_up': self.spun_up, 'escs_armed': self.fly.armed,
            },
            'fire': {'firing': self.firing, 'mode': self.cfg['fire.mode'], 'shots': self.shots,
                     'action': {True: 'fire', False: 'load', None: None}[self.fire_uses_wheels()],
                     'ammo': self.ammo if self.cfg['loader.magazine'] > 0 else None,
                     'magazine': self.cfg['loader.magazine']},
            'heartbeat_age_ms': None if self.last_heartbeat is None
            else round((now - self.last_heartbeat) * 1000),
            'motion': {**self.motion.stats(), 'alive': self.motion.alive,
                       'backend': self.motion.backend},
            'camera': self.camera.status() if self.camera else None,
            'aim': self.aim.status,
            'system': system.snapshot(),
            'restart_needed': sorted(self.restart_needed),
        }
