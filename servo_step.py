#!/usr/bin/env python3
"""
Servo step script for Raspberry Pi — 360° continuous rotation servos.
Each arrow key press rotates the servo by exactly STEP_DEGREES at full speed,
then stops.

Controls:
  Left / Right  →  horizontal servo  (CCW / CW)
  Up   / Down   →  vertical servo    (CW  / CCW)
  Q             →  quit
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
# ─────────────────────────────────────────────────────────────────────────────

# Derived — how long to run at full speed to cover STEP_DEGREES
STEP_DURATION_S: float = STEP_DEGREES * FULL_SPEED_TIME_S / FULL_SPEED_DEGREES

KEY_UP    = '\x1b[A'
KEY_DOWN  = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT  = '\x1b[D'
KEY_QUIT  = 'q'


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def pulse(pwm, duty: float) -> None:
    """Run servo at duty for exactly STEP_DURATION_S, then stop."""
    pwm.ChangeDutyCycle(duty)
    time.sleep(STEP_DURATION_S)
    pwm.ChangeDutyCycle(DUTY_STOP)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_HORIZONTAL_PIN, GPIO.OUT)
    GPIO.setup(SERVO_VERTICAL_PIN,   GPIO.OUT)

    pwm_h = GPIO.PWM(SERVO_HORIZONTAL_PIN, PWM_FREQUENCY)
    pwm_v = GPIO.PWM(SERVO_VERTICAL_PIN,   PWM_FREQUENCY)

    pwm_h.start(DUTY_STOP)
    pwm_v.start(DUTY_STOP)

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down…')
        pwm_h.ChangeDutyCycle(DUTY_STOP)
        pwm_v.ChangeDutyCycle(DUTY_STOP)
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
    print('  Q = quit\n')

    while True:
        key = read_key()

        if key == KEY_LEFT:
            pulse(pwm_h, DUTY_FULL_CCW)
        elif key == KEY_RIGHT:
            pulse(pwm_h, DUTY_FULL_CW)
        elif key == KEY_UP:
            pulse(pwm_v, DUTY_FULL_CW)
        elif key == KEY_DOWN:
            pulse(pwm_v, DUTY_FULL_CCW)
        elif key.lower() == KEY_QUIT:
            shutdown()


if __name__ == '__main__':
    main()
