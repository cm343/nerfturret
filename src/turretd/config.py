"""
Parameter schema and persisted configuration for turretd.

Every tunable value of the turret is declared once in SCHEMA with its type,
unit, range, default and how a change takes effect:

  apply = "live"     takes effect immediately
          "camera"   the camera restarts automatically to apply it
          "restart"  needs a turretd restart (e.g. microstep wiring, ESC range)

The UI builds its config page from this schema (GET /api/schema), so a new
parameter is one Param(...) line here. Values are stored as JSON (only the ones
that differ from the default), written atomically. Named presets are whole
snapshots of the values in a presets/ directory next to the config file.
"""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

DEFAULT_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config'
)

_PRESET_NAME = re.compile(r'^[A-Za-z0-9 _.-]{1,40}$')


@dataclass(frozen=True)
class Param:
    key: str                        # "group.name"
    label: str
    type: str                       # float | int | bool | enum
    default: Any
    unit: str = ''
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: tuple = ()
    apply: str = 'live'
    help: str = ''

    @property
    def group(self) -> str:
        return self.key.split('.', 1)[0]

    def to_dict(self) -> dict:
        d = asdict(self)
        d['choices'] = list(self.choices)
        d['group'] = self.group
        return d


def _axis_params(axis: str, *, speed, accel, jog, lo, hi, hold, help_hold) -> list:
    return [
        Param(f'{axis}.speed', 'Speed', 'float', speed, '°/s', 1, 720, 1,
              help='Normal move speed at the output (after the gear ratio).'),
        Param(f'{axis}.accel', 'Acceleration', 'float', accel, '°/s²', 5, 5000, 5),
        Param(f'{axis}.fast_factor', 'Fast multiplier (Shift)', 'float', 2.5, '×', 1, 10, 0.1),
        Param(f'{axis}.fine_factor', 'Fine multiplier (Alt)', 'float', 0.25, '×', 0.02, 1, 0.01),
        Param(f'{axis}.jog_step', 'Jog step', 'float', jog, '°', 0.05, 90, 0.05,
              help='Distance of one jog-button press.'),
        Param(f'{axis}.limit_min', 'Lower limit', 'float', lo, '°', -360, 360, 0.5,
              help='Soft limit, relative to the zero reference.'),
        Param(f'{axis}.limit_max', 'Upper limit', 'float', hi, '°', -360, 360, 0.5),
        Param(f'{axis}.limits_enabled', 'Enforce limits', 'bool', True,
              help='Limits only apply once the axis is referenced.'),
        Param(f'{axis}.gear_ratio', 'Gear ratio', 'float', 1.0, 'motor rev / output rev',
              0.1, 100, 0.01, help='E.g. 3.0 for a 20T → 60T belt reduction.'),
        Param(f'{axis}.invert', 'Invert direction', 'bool', False),
        Param(f'{axis}.hold_when_idle', 'Hold torque when idle', 'bool', hold, help=help_hold),
        Param(f'{axis}.idle_disable', 'Disable driver after idle', 'float', 30.0, 's', 1, 3600, 1,
              help='Only used when "Hold torque when idle" is off.'),
        Param(f'{axis}.microsteps', 'Microsteps', 'enum', 16, choices=(1, 2, 4, 8, 16, 32),
              apply='restart', help='Must match the M0–M2 wiring on the driver board.'),
    ]


SCHEMA: list[Param] = [
    # ── flywheels ────────────────────────────────────────────────────────────
    Param('flywheel.throttle', 'Throttle', 'float', 30.0, '%', 0, 100, 1),
    Param('flywheel.trim', 'Left/right trim', 'float', 0.0, '%', -20, 20, 0.5,
          help='Positive = left wheel faster. Corrects darts that curve sideways.'),
    Param('flywheel.max_throttle', 'Throttle cap', 'float', 60.0, '%', 5, 100, 1,
          help='Hard upper limit for either wheel, including trim.'),
    Param('flywheel.ramp_rate', 'Spin-up ramp', 'float', 100.0, '%/s', 10, 1000, 10),
    Param('flywheel.spinup_time', 'Spin-up time before firing', 'float', 1.0, 's', 0, 10, 0.1,
          help='Wait at target speed before the loader may push a dart.'),
    Param('flywheel.idle_spindown', 'Auto spin-down after idle', 'float', 30.0, 's', 0, 600, 1,
          help='0 = never.'),
    Param('flywheel.arm_time', 'ESC arming time', 'float', 3.0, 's', 0, 10, 0.5,
          help='Stop pulse held on the first ARM so the ESCs finish arming.'),
    Param('flywheel.esc_min_us', 'ESC stop pulse', 'int', 1000, 'µs', 800, 1200, 10, apply='restart'),
    Param('flywheel.esc_max_us', 'ESC full pulse', 'int', 2000, 'µs', 1800, 2200, 10, apply='restart'),
    Param('flywheel.backend', 'Pulse backend', 'enum', 'lgpio', choices=('lgpio', 'hwpwm'),
          apply='restart', help='hwpwm needs the pwm-2chan overlay in config.txt.'),

    # ── pan / tilt ───────────────────────────────────────────────────────────
    *_axis_params('pan', speed=60.0, accel=180.0, jog=5.0, lo=-90.0, hi=90.0, hold=True,
                  help_hold='Keeps the turret from drifting; costs motor heat.'),
    *_axis_params('tilt', speed=45.0, accel=135.0, jog=2.0, lo=-20.0, hi=45.0, hold=True,
                  help_hold='Keep on: without holding torque the barrel can sag.'),

    # ── loader ───────────────────────────────────────────────────────────────
    Param('loader.stroke', 'Stroke', 'float', 240.0, '° motor', 10, 3600, 1,
          help='Motor rotation from rest to the front (dart pushed into the flywheels).'),
    Param('loader.push_speed', 'Push speed', 'float', 540.0, '°/s', 10, 3600, 10),
    Param('loader.return_speed', 'Return speed', 'float', 540.0, '°/s', 10, 3600, 10),
    Param('loader.accel', 'Acceleration', 'float', 2160.0, '°/s²', 50, 50000, 50),
    Param('loader.front_dwell', 'Dwell at front', 'float', 50.0, 'ms', 0, 1000, 5),
    Param('loader.rate', 'Fire rate', 'float', 3.0, 'darts/s', 0.2, 10, 0.1,
          help='Upper bound; the stroke time may make it slower.'),
    Param('loader.burst', 'Burst size', 'int', 3, 'darts', 2, 20, 1),
    Param('loader.magazine', 'Magazine size', 'int', 12, 'darts', 0, 200, 1,
          help='0 = do not count ammo.'),
    Param('loader.invert', 'Invert direction', 'bool', False),
    Param('loader.hold_when_idle', 'Hold torque when idle', 'bool', False),
    Param('loader.idle_disable', 'Disable driver after idle', 'float', 5.0, 's', 1, 3600, 1),
    Param('loader.assume_rest_on_start', 'Assume rest position at start', 'bool', True,
          help='Every fire cycle ends at rest, so after a clean stop the loader is there.'),
    Param('loader.microsteps', 'Microsteps', 'enum', 8, choices=(1, 2, 4, 8, 16, 32),
          apply='restart', help='Must match the M0–M2 wiring on the driver board.'),

    # ── firing ───────────────────────────────────────────────────────────────
    Param('fire.mode', 'Fire mode', 'enum', 'single', choices=('single', 'burst', 'auto')),
    Param('fire.auto_spinup', 'Spin up on fire', 'bool', True,
          help='ARMED: pressing fire starts the flywheels if they are off. '
               'Off: fire with the flywheels off only loads.'),
    Param('fire.load_when_disarmed', 'Load while disarmed', 'bool', True,
          help='DISARMED: the fire button runs the loader only (LOAD); flywheels stay off.'),
    Param('fire.max_fire_time', 'Max continuous fire', 'float', 10.0, 's', 1, 120, 1),

    # ── camera ───────────────────────────────────────────────────────────────
    Param('camera.resolution', 'Resolution', 'enum', '1280x720',
          choices=('640x480', '1280x720', '1920x1080'), apply='camera'),
    Param('camera.fps', 'Frame rate', 'int', 30, 'fps', 5, 60, 1, apply='camera'),
    Param('camera.quality', 'JPEG quality', 'enum', 'high',
          choices=('low', 'medium', 'high', 'very_high'), apply='camera'),
    Param('camera.hflip', 'Mirror horizontally', 'bool', False, apply='camera'),
    Param('camera.vflip', 'Flip vertically', 'bool', False, apply='camera'),
    Param('camera.crosshair_x', 'Crosshair offset X', 'float', 0.0, '', -1, 1, 0.001,
          help='Where darts land, as a fraction of half the image width (−1 left … 1 right).'),
    Param('camera.crosshair_y', 'Crosshair offset Y', 'float', 0.0, '', -1, 1, 0.001,
          help='Fraction of half the image height (−1 top … 1 bottom).'),
    Param('camera.hfov', 'Horizontal field of view', 'float', 53.5, '°', 10, 180, 0.1,
          help='Camera Module v1 (OV5647): 53.5°; v2 (IMX219): 62.2°. Calibrate for accuracy.'),
    Param('camera.vfov', 'Vertical field of view', 'float', 30.1, '°', 10, 180, 0.1,
          help='OV5647: 41.4° at 4:3, about 30° for 16:9 resolutions.'),

    # ── aim & tracking ───────────────────────────────────────────────────────
    Param('aim.mode', 'Click to aim', 'enum', 'auto', choices=('open', 'auto'),
          help='open: one move computed from the field of view. auto: same move, then '
               'track the clicked spot and correct, when it is trackable.'),
    Param('aim.tolerance', 'Aim tolerance', 'float', 0.1, '°', 0.01, 2, 0.01,
          help='Stop correcting once the spot is this close to the crosshair.'),
    Param('aim.max_iter', 'Max corrections', 'int', 3, '', 1, 10, 1),
    Param('aim.settle', 'Settle time', 'int', 150, 'ms', 0, 2000, 10,
          help='Wait after the turret stops before measuring: vibration and motion blur.'),
    Param('aim.patch', 'Feature size', 'int', 31, 'px', 11, 101, 2,
          help='Template size on the 640 px wide tracking image.'),
    Param('aim.search', 'Search radius', 'float', 25.0, '% width', 5, 60, 1,
          help='How far from the crosshair to look after an open-loop move.'),
    Param('aim.min_texture', 'Min. texture', 'float', 3.0, '', 0, 50, 0.1,
          help='Below this the spot is too plain to track (wall, sky): open loop only.'),
    Param('aim.min_score', 'Min. match score', 'float', 0.7, '', 0.3, 0.99, 0.01,
          help='Correlation of the best match (1 = identical). Lighting and perspective '
               'lower it; the margin check still rejects look-alikes.'),
    Param('aim.min_margin', 'Min. match margin', 'float', 0.1, '', 0, 0.9, 0.01,
          help='Best minus second-best match. Low = repeating pattern, ambiguous.'),
    Param('aim.min_sharp', 'Min. sharpness', 'float', 0.5, '×', 0, 1, 0.05,
          help='Sharpness of the match relative to the reference. Low = blurred.'),
    Param('aim.cal_step', 'FOV calibration step', 'float', 2.0, '°', 0.5, 10, 0.5),
    Param('aim.cal_max_rms', 'FOV fit tolerance', 'float', 0.15, '° rms', 0.01, 2, 0.01,
          help='Reject the calibration when the samples fit the lens model worse.'),

    # ── safety ───────────────────────────────────────────────────────────────
    Param('safety.watchdog_ms', 'Link watchdog', 'int', 500, 'ms', 100, 5000, 50,
          help='No heartbeat from the UI for this long → emergency stop.'),
]

PARAMS: dict[str, Param] = {p.key: p for p in SCHEMA}
GROUP_LABELS = {
    'flywheel': 'Flywheels', 'pan': 'Pan', 'tilt': 'Tilt', 'loader': 'Loader',
    'fire': 'Firing', 'camera': 'Camera', 'aim': 'Aim & tracking', 'safety': 'Safety',
}


class ConfigError(ValueError):
    pass


def coerce(p: Param, value: Any) -> Any:
    """Validate and convert a value for `p`. Numbers are clamped to the range."""
    if p.type == 'bool':
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ('true', 'false', '1', '0', 'on', 'off'):
            return value.lower() in ('true', '1', 'on')
        if value in (0, 1):
            return bool(value)
        raise ConfigError(f'{p.key}: expected true/false, got {value!r}')
    if p.type == 'enum':
        for c in p.choices:
            if value == c or str(value) == str(c):
                return c
        raise ConfigError(f'{p.key}: must be one of {list(p.choices)}, got {value!r}')
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f'{p.key}: expected a number, got {value!r}') from None
    if num != num or num in (float('inf'), float('-inf')):
        raise ConfigError(f'{p.key}: expected a finite number')
    if p.min is not None:
        num = max(p.min, num)
    if p.max is not None:
        num = min(p.max, num)
    return int(round(num)) if p.type == 'int' else num


def _atomic_write_json(path: str, data: Any) -> None:
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp-', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


@dataclass
class ConfigStore:
    """Current parameter values, persisted to <dir>/turret.json."""

    directory: str = DEFAULT_CONFIG_DIR
    values: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = {p.key: p.default for p in SCHEMA}
        self.load_errors: list[str] = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._merge(json.load(f))

    @property
    def path(self) -> str:
        return os.path.join(self.directory, 'turret.json')

    @property
    def preset_dir(self) -> str:
        return os.path.join(self.directory, 'presets')

    def _merge(self, data: dict) -> None:
        """Take known, valid keys from `data`; collect problems instead of failing."""
        for key, value in data.items():
            p = PARAMS.get(key)
            if p is None:
                self.load_errors.append(f'unknown parameter {key!r} ignored')
                continue
            try:
                self.values[key] = coerce(p, value)
            except ConfigError as e:
                self.load_errors.append(str(e))

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default=None) -> Any:
        return self.values.get(key, default)

    def set_many(self, changes: dict) -> dict:
        """Validate all changes first, then apply and save. Returns the applied values."""
        applied = {}
        for key, value in changes.items():
            p = PARAMS.get(key)
            if p is None:
                raise ConfigError(f'unknown parameter {key!r}')
            applied[key] = coerce(p, value)
        self.values.update(applied)
        self.save()
        return applied

    def save(self) -> None:
        changed = {k: v for k, v in self.values.items() if v != PARAMS[k].default}
        _atomic_write_json(self.path, changed)

    def snapshot(self) -> dict:
        return dict(self.values)

    # ── presets ─────────────────────────────────────────────────────────────
    def _preset_path(self, name: str) -> str:
        if not _PRESET_NAME.match(name or ''):
            raise ConfigError('preset names: 1–40 letters, digits, space, _ . -')
        return os.path.join(self.preset_dir, f'{name}.json')

    def list_presets(self) -> list[str]:
        if not os.path.isdir(self.preset_dir):
            return []
        return sorted(f[:-5] for f in os.listdir(self.preset_dir)
                      if f.endswith('.json') and not f.startswith('.'))

    def save_preset(self, name: str) -> None:
        _atomic_write_json(self._preset_path(name), self.snapshot())

    def load_preset(self, name: str) -> dict:
        """Apply a preset. Returns {key: value} of everything that changed."""
        path = self._preset_path(name)
        if not os.path.exists(path):
            raise ConfigError(f'no preset {name!r}')
        with open(path) as f:
            data = json.load(f)
        before = self.snapshot()
        self.load_errors = []
        self._merge(data)
        self.save()
        return {k: v for k, v in self.values.items() if before[k] != v}

    def delete_preset(self, name: str) -> None:
        path = self._preset_path(name)
        if not os.path.exists(path):
            raise ConfigError(f'no preset {name!r}')
        os.unlink(path)


def schema_json() -> dict:
    return {'groups': GROUP_LABELS, 'params': [p.to_dict() for p in SCHEMA]}
