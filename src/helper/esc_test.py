#!/usr/bin/env python3
"""
ESC / flywheel motor test script for Raspberry Pi. Pins follow
nerf-turret-wiring.html (Rev B): ESC 1 (flywheel L) on GPIO 12, ESC 2
(flywheel R) on GPIO 13. 50 Hz servo pulses, 1000 µs = stop, 2000 µs = full.

Pulse backends (see src/esc.py):
  --backend lgpio  (default) lgpio.tx_servo, works without any setup.
  --backend hwpwm  kernel hardware PWM, jitter-free; needs
                   dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4
                   in /boot/firmware/config.txt and a reboot.
If the motors stutter or the speed wanders with lgpio, try hwpwm.

Power-up order:
  1. Run this script with the ESC power switch OFF. It starts sending the
     stop pulse (1000 µs) right away.
  2. Switch the ESC power on when prompted. The script holds the stop pulse
     while the ESCs beep and arm.

Controls:
  1 / 2 / B     →  select ESC 1 (left) / ESC 2 (right) / both
  Up / Down     →  throttle +/- step (ramped)
  ] / [         →  double / halve the throttle step
  Space         →  STOP the selected motor(s) immediately (coast down)
  Esc / Q       →  stop everything and quit
  Ctrl-C        →  stop everything and quit

Throttle is capped by --max-throttle (default 30 %). Raise it only once both
flywheels are mounted, balanced and spin the right way.

--calibrate runs the ESC throttle-range calibration (max then min) instead of
the interactive test. Most BLHeli_S / AM32 ESCs don't need it. Do it with the
motors unloaded and read your ESC's manual first.

Usage:
  python3 src/helper/esc_test.py [--esc 1|2|both] [--max-throttle 0.3] [--backend lgpio|hwpwm]
  python3 src/helper/esc_test.py --calibrate
"""

import argparse
import os
import signal
import sys
import termios
import time
import tty

# Make the src/ package importable when this script is run directly from helper/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esc import ESC, HardwarePwmOutput, LgpioServoOutput  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
ESC_PINS  = {'1': 12, '2': 13}          # BCM: ESC 1 = flywheel L, ESC 2 = flywheel R
ESC_NAMES = {'1': 'left', '2': 'right'}
MIN_US, MAX_US = 1000, 2000

DEFAULT_MAX_THROTTLE = 0.30
START_STEP           = 0.05             # throttle change per Up/Down press
MIN_STEP, MAX_STEP   = 0.01, 0.20
SLEW                 = 0.5              # throttle fraction / s while ramping
ARM_SECONDS          = 3.0
TICK_S               = 0.02             # one 50 Hz frame
# ─────────────────────────────────────────────────────────────────────────────

KEY_UP      = '\x1b[A'
KEY_DOWN    = '\x1b[B'
KEY_ESC     = '\x1b'
KEY_STOP    = ' '
KEY_BIGGER  = ']'
KEY_SMALLER = '['
KEY_BOTH    = 'b'
KEY_QUIT    = 'q'


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


def calibrate(output) -> None:
    """Teach the ESCs the 1000–2000 µs range: max at power-up, then min."""
    print('ESC throttle-range calibration.')
    print('  !! Remove the flywheels or make sure nothing can be thrown. !!')
    input('  ESC power must be OFF. Press Enter to start sending FULL throttle… ')
    for pin in ESC_PINS.values():
        output.set_pulse_us(pin, MAX_US)
    input('  Now switch ESC power ON. Wait for the "max" beeps, then press Enter… ')
    for pin in ESC_PINS.values():
        output.set_pulse_us(pin, MIN_US)
    print('  Sent min. Wait for the confirmation beeps (about 3 s)…')
    time.sleep(4)
    print('  Done. Switch ESC power off and rerun without --calibrate.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Spin the flywheel ESCs from the keyboard.')
    parser.add_argument('--esc', choices=['1', '2', 'both'], default='both',
                        help='ESC(s) selected at start (default: both)')
    parser.add_argument('--max-throttle', type=float, default=DEFAULT_MAX_THROTTLE,
                        help=f'throttle cap 0–1 (default: {DEFAULT_MAX_THROTTLE})')
    parser.add_argument('--backend', choices=['lgpio', 'hwpwm'], default='lgpio',
                        help='pulse generator (default: lgpio)')
    parser.add_argument('--calibrate', action='store_true',
                        help='run ESC throttle-range calibration and exit')
    args = parser.parse_args()
    if not 0.0 < args.max_throttle <= 1.0:
        parser.error('--max-throttle must be in (0, 1]')

    output = LgpioServoOutput() if args.backend == 'lgpio' else HardwarePwmOutput()

    if args.calibrate:
        try:
            calibrate(output)
        finally:
            for pin in ESC_PINS.values():
                output.set_pulse_us(pin, 0)
            output.close()
        return

    escs = {k: ESC(output, pin, min_us=MIN_US, max_us=MAX_US, max_throttle=args.max_throttle)
            for k, pin in ESC_PINS.items()}
    # Stop pulse on both lines before anything else, even if only one is tested.
    for e in escs.values():
        e.setup()

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down… (throttle 0, pulses off)')
        for e in escs.values():
            e.cleanup()
        output.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f'ESC test control ({args.backend}).')
    for k, pin in ESC_PINS.items():
        print(f'  {k} = ESC {k} ({ESC_NAMES[k]})  GPIO {pin}')
    print(f'  Throttle cap: {args.max_throttle:.0%}   ({MIN_US}–{MAX_US} µs)')
    print(f'  Sending stop pulse ({MIN_US} µs) on both ESCs.')
    input('  Switch ESC power ON now, then press Enter to arm… ')
    print(f'  Arming: holding stop for {ARM_SECONDS:.0f} s…')
    time.sleep(ARM_SECONDS)
    for e in escs.values():
        e.arm(0)        # already held stop for ARM_SECONDS above
    print('  Armed.  ↑/↓ throttle   ]/[ step   1/2/B select   Space STOP   Esc/Q quit')
    print()

    selected = ['1', '2'] if args.esc == 'both' else [args.esc]
    step = START_STEP

    def status() -> None:
        sel = 'both' if len(selected) == 2 else f'ESC {selected[0]}'
        thr = '  '.join(f'ESC{k} {e.throttle:4.0%} ({e.pulse_us(e.throttle)} µs)'
                        for k, e in escs.items())
        print(f'  [{sel:<5}] {thr}   step {step:3.0%}   ', end='\r')

    def ramp(target: float) -> None:
        # Ramp the selected ESCs together, one frame at a time.
        while not all([escs[k].ramp_step(target, SLEW * TICK_S) for k in selected]):
            status()
            time.sleep(TICK_S)

    status()

    while True:
        key = read_key()
        current = max(escs[k].throttle for k in selected)

        if key == KEY_UP:
            ramp(current + step)
        elif key == KEY_DOWN:
            ramp(current - step)
        elif key == KEY_STOP:
            for k in selected:
                escs[k].stop()
        elif key == KEY_BIGGER:
            step = min(MAX_STEP, step * 2)
        elif key == KEY_SMALLER:
            step = max(MIN_STEP, step / 2)
        elif key in ESC_PINS:
            selected = [key]
            print(f'\n  Selected ESC {key} ({ESC_NAMES[key]}).')
        elif key.lower() == KEY_BOTH:
            selected = ['1', '2']
            print('\n  Selected both ESCs.')
        elif key == KEY_ESC or key.lower() == KEY_QUIT:
            shutdown()

        status()


if __name__ == '__main__':
    main()
