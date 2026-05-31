#!/usr/bin/env python3
"""
Servo test script for Raspberry Pi.
Control two servos with arrow keys:
  Left / Right  →  horizontal servo
  Up   / Down   →  vertical servo
  Q             →  quit
"""

import sys
import tty
import termios
import signal
import time

import RPi.GPIO as GPIO

# ── Configuration ────────────────────────────────────────────────────────────
SERVO_HORIZONTAL_PIN = 17       # BCM GPIO pin for left/right servo
SERVO_VERTICAL_PIN   = 27       # BCM GPIO pin for up/down servo

PWM_FREQUENCY        = 50       # Hz — standard servo frequency

ANGLE_MIN            = 0        # degrees — hard lower limit
ANGLE_MAX            = 180      # degrees — hard upper limit
ANGLE_START          = 90       # degrees — starting position for both servos
ANGLE_STEP           = 5        # degrees moved per key press

# Pulse-width mapping.  Adjust if your servo doesn't reach its physical limits.
DUTY_MIN             = 2.5      # % duty cycle at ANGLE_MIN (typically 1 ms pulse)
DUTY_MAX             = 12.5     # % duty cycle at ANGLE_MAX (typically 2 ms pulse)

UPDATE_DELAY         = 0.02     # seconds to hold the PWM signal after each move
# ─────────────────────────────────────────────────────────────────────────────

# ANSI escape sequences produced by arrow keys in a standard terminal
KEY_UP    = '\x1b[A'
KEY_DOWN  = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT  = '\x1b[D'
KEY_QUIT  = 'q'


def angle_to_duty(angle: float) -> float:
    """Map an angle in [ANGLE_MIN, ANGLE_MAX] to a PWM duty-cycle percentage."""
    ratio = (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
    return DUTY_MIN + ratio * (DUTY_MAX - DUTY_MIN)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def read_key() -> str:
    """Read one keypress (including multi-byte arrow-key sequences) from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':              # start of escape sequence
            ch += sys.stdin.read(2)   # consume '[' + letter
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_HORIZONTAL_PIN, GPIO.OUT)
    GPIO.setup(SERVO_VERTICAL_PIN,   GPIO.OUT)

    pwm_h = GPIO.PWM(SERVO_HORIZONTAL_PIN, PWM_FREQUENCY)
    pwm_v = GPIO.PWM(SERVO_VERTICAL_PIN,   PWM_FREQUENCY)

    angle_h = float(ANGLE_START)
    angle_v = float(ANGLE_START)

    pwm_h.start(angle_to_duty(angle_h))
    pwm_v.start(angle_to_duty(angle_v))

    def shutdown(sig=None, frame=None) -> None:
        print('\nShutting down…')
        pwm_h.stop()
        pwm_v.stop()
        GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print('Servo test running.')
    print(f'  Horizontal servo  GPIO {SERVO_HORIZONTAL_PIN}  ←/→ arrow keys')
    print(f'  Vertical servo    GPIO {SERVO_VERTICAL_PIN}    ↑/↓ arrow keys')
    print(f'  Range {ANGLE_MIN}°–{ANGLE_MAX}°  •  step {ANGLE_STEP}°  •  start {ANGLE_START}°')
    print('Press Q to quit.\n')

    def status() -> None:
        print(f'  Horizontal: {angle_h:>6.1f}°   Vertical: {angle_v:>6.1f}°', end='\r')

    status()

    while True:
        key = read_key()

        if key == KEY_LEFT:
            angle_h = clamp(angle_h - ANGLE_STEP, ANGLE_MIN, ANGLE_MAX)
            pwm_h.ChangeDutyCycle(angle_to_duty(angle_h))
        elif key == KEY_RIGHT:
            angle_h = clamp(angle_h + ANGLE_STEP, ANGLE_MIN, ANGLE_MAX)
            pwm_h.ChangeDutyCycle(angle_to_duty(angle_h))
        elif key == KEY_UP:
            angle_v = clamp(angle_v + ANGLE_STEP, ANGLE_MIN, ANGLE_MAX)
            pwm_v.ChangeDutyCycle(angle_to_duty(angle_v))
        elif key == KEY_DOWN:
            angle_v = clamp(angle_v - ANGLE_STEP, ANGLE_MIN, ANGLE_MAX)
            pwm_v.ChangeDutyCycle(angle_to_duty(angle_v))
        elif key.lower() == KEY_QUIT:
            shutdown()

        time.sleep(UPDATE_DELAY)
        status()


if __name__ == '__main__':
    main()
