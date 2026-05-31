#!/usr/bin/env python3
"""
Servo — a 360° continuous-rotation servo on a Raspberry Pi.

A continuous-rotation servo has no position feedback: its PWM duty cycle sets
*speed and direction*, not an absolute angle. This class therefore offers:

  * set_speed(speed)        — free-run at a signed speed fraction
  * move_by(degrees, speed) — relative timed rotation, computed from the
                              servo's known full-speed rate (deg_per_sec)

There is no absolute positioning; move_by is open-loop (timing based).
"""

import time

import RPi.GPIO as GPIO

# Default duty-cycle calibration (percent of the 20 ms / 50 Hz period).
# DUTY_STOP is the trimmed neutral that produces no drift.
DEFAULT_PWM_FREQUENCY = 50      # Hz
DEFAULT_DUTY_STOP     = 7.1     # % — neutral / stopped
DEFAULT_DUTY_FULL_CW  = 12.5    # % — full clockwise speed
DEFAULT_DUTY_FULL_CCW = 2.5     # % — full counter-clockwise speed


class Servo:
    """A single 360° continuous-rotation servo driven by GPIO PWM."""

    def __init__(
        self,
        pin: int,
        deg_per_sec: float,
        *,
        pwm_frequency: int = DEFAULT_PWM_FREQUENCY,
        duty_stop: float = DEFAULT_DUTY_STOP,
        duty_full_cw: float = DEFAULT_DUTY_FULL_CW,
        duty_full_ccw: float = DEFAULT_DUTY_FULL_CCW,
    ) -> None:
        """
        :param pin:           BCM GPIO pin the servo signal wire is connected to.
        :param deg_per_sec:   rotation rate at full speed, in degrees/second
                              (e.g. 60° in 0.2 s -> 300.0).
        :param pwm_frequency: PWM frequency in Hz (50 Hz is standard).
        :param duty_stop:     duty cycle (%) at which the servo is stationary.
        :param duty_full_cw:  duty cycle (%) for full clockwise speed.
        :param duty_full_ccw: duty cycle (%) for full counter-clockwise speed.
        """
        if deg_per_sec <= 0:
            raise ValueError("deg_per_sec must be positive")

        self.pin = pin
        self.deg_per_sec = float(deg_per_sec)
        self.pwm_frequency = pwm_frequency
        self.duty_stop = duty_stop
        self.duty_full_cw = duty_full_cw
        self.duty_full_ccw = duty_full_ccw

        self._pwm = None  # created in start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Configure the GPIO pin and begin emitting the stop pulse.

        Assumes GPIO.setmode(GPIO.BCM) has already been set by the caller, or
        sets it here if not yet configured.
        """
        if GPIO.getmode() is None:
            GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self._pwm = GPIO.PWM(self.pin, self.pwm_frequency)
        self._pwm.start(self.duty_stop)

    def stop(self) -> None:
        """Hold the servo stationary at the neutral duty cycle."""
        self._require_started()
        self._pwm.ChangeDutyCycle(self.duty_stop)

    def cleanup(self) -> None:
        """Stop this servo's PWM. Does NOT call GPIO.cleanup() — leave that to
        the owner so multiple servos can share the GPIO subsystem."""
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(self.duty_stop)
            time.sleep(0.05)
            self._pwm.stop()
            self._pwm = None

    # ── motion ───────────────────────────────────────────────────────────────
    def set_speed(self, speed: float) -> None:
        """Free-run at a signed speed fraction.

        :param speed: in [-1.0, 1.0]. 0 stops, >0 is clockwise, <0 is
                      counter-clockwise. Magnitude scales linearly between the
                      stop duty and the full-speed duty for that direction.
        """
        self._require_started()
        self._pwm.ChangeDutyCycle(self._speed_to_duty(speed))

    def move_by(self, degrees: float, speed: float = 1.0) -> None:
        """Rotate (open-loop) by a relative number of degrees, then stop.

        Blocking: sleeps for the computed duration. Run from a worker thread if
        you need it off the main thread (see Trigger).

        :param degrees: signed amount to rotate. >0 clockwise, <0 CCW.
        :param speed:   speed magnitude fraction in (0, 1].
        """
        self._require_started()
        speed = abs(speed)
        if speed == 0:
            return
        if speed > 1.0:
            speed = 1.0
        if degrees == 0:
            return

        duration = abs(degrees) / (self.deg_per_sec * speed)
        direction = 1.0 if degrees > 0 else -1.0
        self.set_speed(direction * speed)
        time.sleep(duration)
        self.stop()

    # ── internals ─────────────────────────────────────────────────────────────
    def _speed_to_duty(self, speed: float) -> float:
        """Map a signed speed fraction in [-1, 1] to a duty-cycle percentage."""
        if speed > 1.0:
            speed = 1.0
        elif speed < -1.0:
            speed = -1.0

        if speed == 0:
            return self.duty_stop
        if speed > 0:
            return self.duty_stop + speed * (self.duty_full_cw - self.duty_stop)
        return self.duty_stop + speed * (self.duty_stop - self.duty_full_ccw)

    def _require_started(self) -> None:
        if self._pwm is None:
            raise RuntimeError("Servo.start() must be called before moving the servo")
