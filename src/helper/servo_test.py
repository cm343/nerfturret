#!/usr/bin/env python3
"""
Servo test script for Raspberry Pi — 360° continuous rotation servos.
PWM duty cycle controls speed and direction (not angle).

Controls:
  Left / Right  →  horizontal servo  (CCW / CW)
  Up   / Down   →  vertical servo    (CW  / CCW)
  Space         →  stop both servos
  Q             →  quit
"""

import sys
import tty
import termios
import signal
import time

import RPi.GPIO as GPIO

# ── Configuration ────────────────────────────────────────────────────────────
SERVO_HORIZONTAL_PIN = 17       # BCM GPIO pin for horizontal servo
SERVO_VERTICAL_PIN   = 27       # BCM GPIO pin for vertical servo

PWM_FREQUENCY        = 50       # Hz — standard servo frequency

# Duty-cycle calibration.  At DUTY_STOP the servo should be stationary.
# Adjust DUTY_STOP first (trim), then the CW/CCW limits if needed.
DUTY_STOP            = 7.1     # % — neutral / stopped (trimmed, no drift)
DUTY_FULL_CW         = 12.5    # % — maximum clockwise speed  (~2 ms pulse)
DUTY_FULL_CCW        = 2.5     # % — maximum counter-clockwise speed  (~1 ms pulse)

# Number of discrete speed steps between stopped and full speed.
SPEED_STEPS          = 10

# Seconds between reading the next key (keep short so the servo reacts quickly).
UPDATE_DELAY         = 0.05
# ─────────────────────────────────────────────────────────────────────────────

# ANSI escape sequences produced by arrow keys in a standard terminal
KEY_UP    = '\x1b[A'
KEY_DOWN  = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT  = '\x1b[D'
KEY_STOP  = ' '
KEY_QUIT  = 'q'


def speed_to_duty(speed: int) -> float:
    """
    Convert a speed value in [-SPEED_STEPS, +SPEED_STEPS] to a duty cycle.
    0 → DUTY_STOP, positive → CW, negative → CCW.
    """
    if speed == 0:
        return DUTY_STOP
    if speed > 0:
        return DUTY_STOP + speed * (DUTY_FULL_CW  - DUTY_STOP) / SPEED_STEPS
    return     DUTY_STOP + speed * (DUTY_STOP - DUTY_FULL_CCW) / SPEED_STEPS


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def read_key() -> str:
    """Read one keypress (including multi-byte arrow-key sequences) from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def speed_bar(speed: int, width: int = SPEED_STEPS) -> str:
    """Simple ASCII bar: negative fills left of center, positive fills right."""
    bar = ['-'] * (2 * width + 1)
    bar[width] = '|'
    if speed > 0:
        for i in range(1, speed + 1):
            bar[width + i] = '>'
    elif speed < 0:
        for i in range(1, -speed + 1):
            bar[width - i] = '<'
    return ''.join(bar)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_HORIZONTAL_PIN, GPIO.OUT)
    GPIO.setup(SERVO_VERTICAL_PIN,   GPIO.OUT)

    pwm_h = GPIO.PWM(SERVO_HORIZONTAL_PIN, PWM_FREQUENCY)
    pwm_v = GPIO.PWM(SERVO_VERTICAL_PIN,   PWM_FREQUENCY)

    speed_h = 0
    speed_v = 0

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

    print('360° Servo test running.')
    print(f'  Horizontal  GPIO {SERVO_HORIZONTAL_PIN}  ←/→  (CCW / CW)')
    print(f'  Vertical    GPIO {SERVO_VERTICAL_PIN}    ↑/↓  (CW  / CCW)')
    print(f'  Speed steps: {SPEED_STEPS}   Stop duty: {DUTY_STOP}%')
    print('  Space = stop both   Q = quit\n')

    def status() -> None:
        print(
            f'  H {speed_bar(speed_h)}  {speed_h:+d}   '
            f'V {speed_bar(speed_v)}  {speed_v:+d}   ',
            end='\r',
        )

    status()

    while True:
        key = read_key()

        if key == KEY_LEFT:
            speed_h = clamp(speed_h - 1, -SPEED_STEPS, SPEED_STEPS)
            pwm_h.ChangeDutyCycle(speed_to_duty(speed_h))
        elif key == KEY_RIGHT:
            speed_h = clamp(speed_h + 1, -SPEED_STEPS, SPEED_STEPS)
            pwm_h.ChangeDutyCycle(speed_to_duty(speed_h))
        elif key == KEY_UP:
            speed_v = clamp(speed_v + 1, -SPEED_STEPS, SPEED_STEPS)
            pwm_v.ChangeDutyCycle(speed_to_duty(speed_v))
        elif key == KEY_DOWN:
            speed_v = clamp(speed_v - 1, -SPEED_STEPS, SPEED_STEPS)
            pwm_v.ChangeDutyCycle(speed_to_duty(speed_v))
        elif key == KEY_STOP:
            speed_h = 0
            speed_v = 0
            pwm_h.ChangeDutyCycle(DUTY_STOP)
            pwm_v.ChangeDutyCycle(DUTY_STOP)
        elif key.lower() == KEY_QUIT:
            shutdown()

        time.sleep(UPDATE_DELAY)
        status()


if __name__ == '__main__':
    main()
