#!/usr/bin/env python3
"""
Servo step script for Raspberry Pi — 360° continuous rotation servos.
Each arrow key press rotates the servo by exactly STEP_DEGREES at full speed,
then stops. Pan/tilt are driven through the shared Servo class (src/servo.py);
the trigger is driven through the Trigger module (src/trigger.py).

Controls:
  Left / Right  →  horizontal servo  (CCW / CW)
  Up   / Down   →  vertical servo    (CW  / CCW)
  T             →  fire the trigger servo (Trigger module, dedicated pin)
  + / -         →  trim DUTY_STOP up / down by TRIM_STEP (both servos)
  P             →  print current DUTY_STOP so you can copy it into config
  Esc / Q       →  quit
"""

import os
import sys
import tty
import termios
import signal

import RPi.GPIO as GPIO

# Make the src/ package importable when this script is run directly from helper/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servo import Servo      # noqa: E402
from trigger import Trigger  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
SERVO_HORIZONTAL_PIN  = 17      # BCM GPIO pin for horizontal servo
SERVO_VERTICAL_PIN    = 27      # BCM GPIO pin for vertical servo

PWM_FREQUENCY         = 50      # Hz — standard servo frequency (20 ms period)

DUTY_STOP             = 7.1     # % — neutral / stopped (trimmed, no drift)
DUTY_FULL_CW          = 12.5    # % — full clockwise speed  (~2 ms pulse)
DUTY_FULL_CCW         = 2.5     # % — full counter-clockwise speed  (~1 ms pulse)

# Measured rotation speed at full speed (use these two to calibrate)
FULL_SPEED_DEGREES    = 60.0    # degrees covered in …
FULL_SPEED_TIME_S     = 0.2     # … this many seconds

STEP_DEGREES          = 10.0    # degrees to rotate per key press

TRIM_STEP             = 0.05    # % duty-cycle change per +/- trim key press
TRIM_MIN              = 5.0     # % — lower bound for DUTY_STOP trim
TRIM_MAX              = 10.0    # % — upper bound for DUTY_STOP trim
# ─────────────────────────────────────────────────────────────────────────────
# The trigger servo is handled by the Trigger module (its own pin & constants).

# Derived — full-speed rate the Servo class needs to time its move_by() rotations
DEG_PER_SEC = FULL_SPEED_DEGREES / FULL_SPEED_TIME_S

KEY_UP         = '\x1b[A'
KEY_DOWN       = '\x1b[B'
KEY_RIGHT      = '\x1b[C'
KEY_LEFT       = '\x1b[D'
KEY_ESC        = '\x1b'
KEY_TRIGGER    = 't'
KEY_TRIM_UP    = '+'
KEY_TRIM_DOWN  = '-'
KEY_PRINT_TRIM = 'p'
KEY_QUIT       = 'q'


def make_servo(pin: int) -> Servo:
    """Build a Servo on `pin` using this script's calibration constants."""
    return Servo(
        pin,
        DEG_PER_SEC,
        pwm_frequency=PWM_FREQUENCY,
        duty_stop=DUTY_STOP,
        duty_full_cw=DUTY_FULL_CW,
        duty_full_ccw=DUTY_FULL_CCW,
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
    GPIO.setmode(GPIO.BCM)

    servo_h = make_servo(SERVO_HORIZONTAL_PIN)
    servo_v = make_servo(SERVO_VERTICAL_PIN)
    servo_h.start()
    servo_v.start()

    duty_stop = DUTY_STOP  # live-trimmed value, mirrored onto both servos

    # Dedicated trigger servo, driven by the Trigger module on its own pin.
    trigger = Trigger()
    trigger.start()

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down…')
        trigger.stop()
        servo_h.cleanup()
        servo_v.cleanup()
        GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    step_ms = STEP_DEGREES / DEG_PER_SEC * 1000
    print('Servo step control.')
    print(f'  Step: {STEP_DEGREES}°  →  {step_ms:.1f} ms pulse at full speed')
    print(f'  Horizontal  GPIO {SERVO_HORIZONTAL_PIN}  ←/→')
    print(f'  Vertical    GPIO {SERVO_VERTICAL_PIN}    ↑/↓')
    print('  T = fire trigger servo (Trigger module)')
    print('  +/- = trim stop point   P = print trim   Esc/Q = quit\n')

    def status() -> None:
        print(f'  DUTY_STOP: {duty_stop:.3f}%   ', end='\r')

    def apply_trim(new_stop: float) -> None:
        """Update the live neutral on both servos and hold there."""
        servo_h.duty_stop = new_stop
        servo_v.duty_stop = new_stop
        servo_h.stop()
        servo_v.stop()

    status()

    while True:
        key = read_key()

        if key == KEY_LEFT:
            servo_h.move_by(-STEP_DEGREES)   # CCW
        elif key == KEY_RIGHT:
            servo_h.move_by(STEP_DEGREES)    # CW
        elif key == KEY_UP:
            servo_v.move_by(STEP_DEGREES)    # CW
        elif key == KEY_DOWN:
            servo_v.move_by(-STEP_DEGREES)   # CCW
        elif key.lower() == KEY_TRIGGER:
            trigger.fire()  # non-blocking; the Trigger worker thread sweeps
        elif key == KEY_TRIM_UP:
            duty_stop = min(TRIM_MAX, round(duty_stop + TRIM_STEP, 4))
            apply_trim(duty_stop)
        elif key == KEY_TRIM_DOWN:
            duty_stop = max(TRIM_MIN, round(duty_stop - TRIM_STEP, 4))
            apply_trim(duty_stop)
        elif key.lower() == KEY_PRINT_TRIM:
            print(f'\n  >>> DUTY_STOP = {duty_stop:.4f}')
        elif key == KEY_ESC or key.lower() == KEY_QUIT:
            shutdown()

        status()


if __name__ == '__main__':
    main()
