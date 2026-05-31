#!/usr/bin/env python3
"""
Trigger — a background worker that fires a fixed servo "shoot" gesture.

A worker thread blocks on a queue.Queue. Calling fire() from any thread enqueues
a signal; the worker performs the sweep: rotate SWEEP_DEGREES one way at full
speed, then SWEEP_DEGREES back. Each fire() is queued (not coalesced), so rapid
triggers all run, one after another.

Run directly on a Raspberry Pi for a smoke test:

    python trigger.py
"""

import queue
import threading
import time

import RPi.GPIO as GPIO

try:
    from .servo import Servo
except ImportError:  # allows running directly from within src/
    from servo import Servo

# ── Configuration ────────────────────────────────────────────────────────────
TRIGGER_PIN      = 22       # BCM GPIO pin for the dedicated trigger servo
DEG_PER_SEC      = 300.0    # full-speed rate: 60° in 0.2 s
PWM_FREQUENCY    = 50       # Hz

DUTY_STOP        = 7.1      # % — trimmed neutral (no drift)
DUTY_FULL_CW     = 12.5     # % — full clockwise speed
DUTY_FULL_CCW    = 2.5      # % — full counter-clockwise speed

SWEEP_DEGREES    = 30.0     # rotate this far one way, then back
SWEEP_SPEED      = 1.0      # speed fraction for the sweep (full speed)
# ─────────────────────────────────────────────────────────────────────────────


class Trigger:
    """Threaded trigger that sweeps a servo 30° out and back on each fire()."""

    _STOP = object()  # sentinel enqueued to tell the worker to exit

    def __init__(
        self,
        *,
        pin: int = TRIGGER_PIN,
        deg_per_sec: float = DEG_PER_SEC,
        sweep_degrees: float = SWEEP_DEGREES,
        sweep_speed: float = SWEEP_SPEED,
    ) -> None:
        self.sweep_degrees = sweep_degrees
        self.sweep_speed = sweep_speed

        self.servo = Servo(
            pin,
            deg_per_sec,
            pwm_frequency=PWM_FREQUENCY,
            duty_stop=DUTY_STOP,
            duty_full_cw=DUTY_FULL_CW,
            duty_full_ccw=DUTY_FULL_CCW,
        )
        self._queue: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Power up the servo and start the worker thread."""
        if self._started:
            return
        self.servo.start()
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        """Signal the worker to exit, join it, and stop this servo's PWM.

        Does NOT call the global GPIO.cleanup() — leave that to the owner so a
        Trigger can be embedded in a script that drives other servos."""
        if not self._started:
            return
        self._queue.put(self._STOP)
        self._thread.join()
        self.servo.cleanup()
        self._started = False

    # ── signalling ──────────────────────────────────────────────────────────
    def fire(self) -> None:
        """Inter-thread signal: enqueue one sweep. Non-blocking."""
        self._queue.put(True)

    # ── worker ─────────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                self._sweep()
            finally:
                self._queue.task_done()

    def _sweep(self) -> None:
        """Rotate out by sweep_degrees, then back to start."""
        self.servo.move_by(-self.sweep_degrees, self.sweep_speed)
        self.servo.move_by(self.sweep_degrees, self.sweep_speed)

    # ── context manager ────────────────────────────────────────────────────────
    def __enter__(self) -> "Trigger":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def main() -> None:
    """Smoke test: fire one sweep, then shut down cleanly."""
    with Trigger() as trigger:
        print("Trigger armed. Firing one sweep…")
        trigger.fire()
        # Give the worker time to complete the (blocking) sweep before stop()
        # joins the thread. Two moves at full speed ~ 2 * 30/300 s = 0.2 s.
        time.sleep(1.0)
        print("Done.")
    GPIO.cleanup()  # the standalone script owns the GPIO subsystem


if __name__ == "__main__":
    main()
