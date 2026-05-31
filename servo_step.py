#!/usr/bin/env python3
"""
Servo step script for Raspberry Pi — 360° continuous rotation servos.
Each arrow key press rotates the servo by exactly STEP_DEGREES at full speed,
then stops.

Controls:
  Left / Right  →  horizontal servo  (CCW / CW)
  Up   / Down   →  vertical servo    (CW  / CCW)
  + / -         →  trim DUTY_STOP up / down by TRIM_STEP (both servos)
  P             →  print current DUTY_STOP so you can copy it into config
  Esc / Q       →  quit
"""

import sys
import tty
import termios
import signal
import time

import RPi.GPIO as GPIO

# ── Configuration ────────────────────────────────────────────────────────────
SERVO_HORIZONTAL_PIN  = 17      # BCM GPIO pin for horizontal servo
SERVO_VERTICAL_PIN    = 27      # BCM GPIO pin for vertical servo

PWM_FREQUENCY         = 50      # Hz — standard servo frequency (20 ms period)

DUTY_STOP             = 7.5     # % — neutral / stopped  (~1.5 ms pulse)
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

# Derived — how long to run at full speed to cover STEP_DEGREES
STEP_DURATION_S: float = STEP_DEGREES * FULL_SPEED_TIME_S / FULL_SPEED_DEGREES

KEY_UP         = '\x1b[A'
KEY_DOWN       = '\x1b[B'
KEY_RIGHT      = '\x1b[C'
KEY_LEFT       = '\x1b[D'
KEY_ESC        = '\x1b'
KEY_TRIM_UP    = '+'
KEY_TRIM_DOWN  = '-'
KEY_PRINT_TRIM = 'p'
KEY_QUIT       = 'q'


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


def pulse(pwm, duty: float, stop: float) -> None:
    """Run servo at duty for exactly STEP_DURATION_S, then stop."""
    pwm.ChangeDutyCycle(duty)
    time.sleep(STEP_DURATION_S)
    pwm.ChangeDutyCycle(stop)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_HORIZONTAL_PIN, GPIO.OUT)
    GPIO.setup(SERVO_VERTICAL_PIN,   GPIO.OUT)

    pwm_h = GPIO.PWM(SERVO_HORIZONTAL_PIN, PWM_FREQUENCY)
    pwm_v = GPIO.PWM(SERVO_VERTICAL_PIN,   PWM_FREQUENCY)

    duty_stop = DUTY_STOP  # live-trimmed value

    pwm_h.start(duty_stop)
    pwm_v.start(duty_stop)

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down…')
        pwm_h.ChangeDutyCycle(duty_stop)
        pwm_v.ChangeDutyCycle(duty_stop)
        time.sleep(0.1)
        pwm_h.stop()
        pwm_v.stop()
        GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print('Servo step control.')
    print(f'  Step: {STEP_DEGREES}°  →  {STEP_DURATION_S * 1000:.1f} ms pulse at full speed')
    print(f'  Horizontal  GPIO {SERVO_HORIZONTAL_PIN}  ←/→')
    print(f'  Vertical    GPIO {SERVO_VERTICAL_PIN}    ↑/↓')
    print('  +/- = trim stop point   P = print trim   Esc/Q = quit\n')

    def status() -> None:
        print(f'  DUTY_STOP: {duty_stop:.3f}%   ', end='\r')

    status()

    while True:
        key = read_key()

        if key == KEY_LEFT:
            pulse(pwm_h, DUTY_FULL_CCW, duty_stop)
        elif key == KEY_RIGHT:
            pulse(pwm_h, DUTY_FULL_CW, duty_stop)
        elif key == KEY_UP:
            pulse(pwm_v, DUTY_FULL_CW, duty_stop)
        elif key == KEY_DOWN:
            pulse(pwm_v, DUTY_FULL_CCW, duty_stop)
        elif key == KEY_TRIM_UP:
            duty_stop = min(TRIM_MAX, round(duty_stop + TRIM_STEP, 4))
            pwm_h.ChangeDutyCycle(duty_stop)
            pwm_v.ChangeDutyCycle(duty_stop)
        elif key == KEY_TRIM_DOWN:
            duty_stop = max(TRIM_MIN, round(duty_stop - TRIM_STEP, 4))
            pwm_h.ChangeDutyCycle(duty_stop)
            pwm_v.ChangeDutyCycle(duty_stop)
        elif key.lower() == KEY_PRINT_TRIM:
            print(f'\n  >>> DUTY_STOP = {duty_stop:.4f}')
        elif key == KEY_ESC or key.lower() == KEY_QUIT:
            shutdown()

        status()


if __name__ == '__main__':
    main()
