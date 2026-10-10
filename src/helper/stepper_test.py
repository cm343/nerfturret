#!/usr/bin/env python3
"""
Stepper test script for Raspberry Pi — HW-216 / DRV8825 drivers.
Each arrow key press jogs the selected stepper by the current jog size, with a
short accel/decel ramp. Pins follow nerf-turret-wiring.html (Rev C).

Controls:
  Tab / D       →  switch to the next driver (pan → tilt → loader → pan)
  1 / 2 / 3     →  select pan / tilt / loader driver directly
  Left / Right  →  jog backward / forward  (forward = DIR high)
  Up   / Down   →  double / halve the jog size
  ] / [         →  speed up / down (×1.25)
  E             →  toggle driver enable (disabled = coils off, shaft turns freely)
  Z             →  zero the position counter
  Esc / Q       →  quit (disables all drivers)
  Ctrl-C        →  abort a running move and quit

There are no endstops: nothing stops a jog at the end of an axis's travel, so
start with small jogs and watch the mechanism.

Usage:
  python3 src/helper/stepper_test.py [--driver pan|tilt|loader]
"""

import argparse
import os
import sys
import tty
import termios
import signal

import RPi.GPIO as GPIO

# Make the src/ package importable when this script is run directly from helper/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stepper import Stepper  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
# name: (STEP, DIR, EN) BCM pins and the microstep mode wired on M0–M2
DRIVERS = {
    'pan':    dict(step_pin=16, dir_pin=20, en_pin=6, microsteps=16),
    'tilt':   dict(step_pin=23, dir_pin=24, en_pin=5, microsteps=16),
    'loader': dict(step_pin=17, dir_pin=27, en_pin=4, microsteps=8),
}
DRIVER_KEYS = {'1': 'pan', '2': 'tilt', '3': 'loader'}

# name: starting cruise speed (rev/s) and ramp (rev/s²). Pan/tilt run slow
# because the pancake motors lose torque with speed; the loader runs fast.
MOTION = {
    'pan':    dict(speed_rps=0.2, accel_rps2=1.0),
    'tilt':   dict(speed_rps=0.2, accel_rps2=1.0),
    'loader': dict(speed_rps=1.5, accel_rps2=6.0),
}

START_JOG_DEGREES     = 45.0    # initial jog size, at the output shaft
MIN_JOG_STEPS         = 1
MAX_SPEED_RPS         = 3.0     # Python busy-wait timing gets jittery beyond this
SPEED_FACTOR          = 1.25
# ─────────────────────────────────────────────────────────────────────────────

KEY_UP         = '\x1b[A'
KEY_DOWN       = '\x1b[B'
KEY_RIGHT      = '\x1b[C'
KEY_LEFT       = '\x1b[D'
KEY_ESC        = '\x1b'
KEY_FASTER     = ']'
KEY_SLOWER     = '['
KEY_ENABLE     = 'e'
KEY_ZERO       = 'z'
KEY_QUIT       = 'q'
KEY_NEXT_DRIVER = ('\t', 'd')


def make_stepper(name: str) -> Stepper:
    cfg = DRIVERS[name]
    motion = MOTION[name]
    steps_per_rev = 200 * cfg['microsteps']
    return Stepper(
        **cfg,
        speed=motion['speed_rps'] * steps_per_rev,
        accel=motion['accel_rps2'] * steps_per_rev,
        start_speed=0.1 * steps_per_rev,
    )


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            # Peek for a following '[' to distinguish Esc from arrow keys
            fd_flags = termios.tcgetattr(fd)
            fd_flags[6][termios.VMIN]  = 0  # non-blocking
            fd_flags[6][termios.VTIME] = 1  # 100 ms timeout
            termios.tcsetattr(fd, termios.TCSADRAIN, fd_flags)
            rest = sys.stdin.read(2)
            ch = ch + rest if rest else ch
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    parser = argparse.ArgumentParser(description='Jog the turret steppers from the keyboard.')
    parser.add_argument('--driver', choices=DRIVERS, default='loader',
                        help='driver selected at start (default: loader)')
    args = parser.parse_args()

    GPIO.setmode(GPIO.BCM)

    # Set up every driver so all EN lines are driven HIGH (disabled), not just
    # the one under test.
    steppers = {name: make_stepper(name) for name in DRIVERS}
    for s in steppers.values():
        s.setup()

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down… (all drivers disabled)')
        for s in steppers.values():
            s.cleanup()
        GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    current = args.driver
    jog_deg = {name: START_JOG_DEGREES for name in DRIVERS}

    print('Stepper test control.')
    for key, name in DRIVER_KEYS.items():
        c = DRIVERS[name]
        print(f'  {key} = {name:<6}  STEP {c["step_pin"]:>2}  DIR {c["dir_pin"]:>2}  '
              f'EN {c["en_pin"]}  1/{c["microsteps"]} step')
    print('  Tab/D next driver   ←/→ jog   ↑/↓ jog size   ]/[ speed')
    print('  E enable   Z zero   Esc/Q quit')
    print('  No endstops: start with small jogs.\n')

    def status() -> None:
        s = steppers[current]
        deg = s.position * 360.0 / s.steps_per_rev
        rps = s.speed / s.steps_per_rev
        state = 'ON ' if s.enabled else 'off'
        print(f'  [{current:<6}] en {state}  pos {s.position:>7} µst ({deg:8.1f}°)  '
              f'jog {jog_deg[current]:6.1f}°  speed {rps:4.2f} rev/s   ', end='\r')

    def jog(direction: int) -> None:
        s = steppers[current]
        steps = max(MIN_JOG_STEPS, round(jog_deg[current] / 360.0 * s.steps_per_rev))
        s.move(direction * steps)

    def set_current(name: str) -> None:
        nonlocal current
        if name != current:
            # Only one driver energised at a time while testing.
            steppers[current].disable()
            current = name
            c = DRIVERS[name]
            print(f'\n  Selected {name}: STEP {c["step_pin"]}  DIR {c["dir_pin"]}  '
                  f'EN {c["en_pin"]}  1/{c["microsteps"]} step '
                  f'(previous driver disabled)')

    status()

    while True:
        key = read_key()
        s = steppers[current]

        if key == KEY_RIGHT:
            jog(+1)
        elif key == KEY_LEFT:
            jog(-1)
        elif key == KEY_UP:
            jog_deg[current] = min(360.0 * 10, jog_deg[current] * 2)
        elif key == KEY_DOWN:
            jog_deg[current] = max(360.0 / s.steps_per_rev, jog_deg[current] / 2)
        elif key == KEY_FASTER:
            s.speed = min(MAX_SPEED_RPS * s.steps_per_rev, s.speed * SPEED_FACTOR)
        elif key == KEY_SLOWER:
            s.speed = max(s.start_speed, s.speed / SPEED_FACTOR)
        elif key.lower() == KEY_ENABLE:
            s.disable() if s.enabled else s.enable()
        elif key.lower() == KEY_ZERO:
            s.zero()
        elif key in DRIVER_KEYS:
            set_current(DRIVER_KEYS[key])
        elif key.lower() in KEY_NEXT_DRIVER:
            names = list(DRIVERS)
            set_current(names[(names.index(current) + 1) % len(names)])
        elif key == KEY_ESC or key.lower() == KEY_QUIT:
            shutdown()

        status()


if __name__ == '__main__':
    main()
