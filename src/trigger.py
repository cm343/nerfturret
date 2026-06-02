#!/usr/bin/env python3
"""
Trigger — a background worker that fires a fixed servo "shoot" gesture.

A worker thread blocks on a queue.Queue. Calling fire() from any thread enqueues
a signal; the worker performs the pull: drive a standard 180° servo from
REST_ANGLE to PULL_ANGLE, pause, then back to REST_ANGLE. Each fire() is queued
(not coalesced), so rapid triggers all run, one after another.

Run directly on a Raspberry Pi for a smoke test:

    python trigger.py
"""

import queue
import threading
import time

import RPi.GPIO as GPIO

try:
    from .servo import PositionalServo
except ImportError:  # allows running directly from within src/
    from servo import PositionalServo

# ── Configuration ────────────────────────────────────────────────────────────
TRIGGER_PIN      = 22       # BCM GPIO pin for the dedicated trigger servo
DEG_PER_SEC      = 150   # full-speed rate: 60° in 0.2 s
PWM_FREQUENCY    = 50       # Hz

DUTY_MIN         = 2.5      # % — duty at 0°   (standard-servo calibration)
DUTY_MAX         = 12.5     # % — duty at 180°

REST_ANGLE       = 0.0      # resting position of the trigger arm
PULL_ANGLE       = 45.0     # angle the arm pulls to, then returns from
PULL_SPEED       = 1.0      # speed fraction for the pull (full speed)
PULL_PAUSE_S     = 0.3      # dwell at the pull position before returning, so the
                            # servo registers the reversal (else it may not move
                            # back when the two moves run back-to-back)
# ─────────────────────────────────────────────────────────────────────────────


class Trigger:
    """Threaded trigger that pulls a standard servo to PULL_ANGLE and back on fire()."""

    _STOP = object()  # sentinel enqueued to tell the worker to exit

    def __init__(
        self,
        *,
        pin: int = TRIGGER_PIN,
        deg_per_sec: float = DEG_PER_SEC,
        rest_angle: float = REST_ANGLE,
        pull_angle: float = PULL_ANGLE,
        pull_speed: float = PULL_SPEED,
        pull_pause: float = PULL_PAUSE_S,
    ) -> None:
        self.rest_angle = rest_angle
        self.pull_angle = pull_angle
        self.pull_speed = pull_speed
        self.pull_pause = pull_pause

        self.servo = PositionalServo(
            pin,
            deg_per_sec,
            pwm_frequency=PWM_FREQUENCY,
            duty_min=DUTY_MIN,
            duty_max=DUTY_MAX,
            initial_angle=rest_angle,
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
                self._pull()
            finally:
                self._queue.task_done()

    def _pull(self) -> None:
        """Drive to the pull angle, pause, then return to the rest angle."""
        self.servo.move_to(self.pull_angle, self.pull_speed)
        time.sleep(self.pull_pause)  # let the servo settle before reversing
        self.servo.move_to(self.rest_angle, self.pull_speed)

    # ── context manager ────────────────────────────────────────────────────────
    def __enter__(self) -> "Trigger":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def main() -> None:
    """Smoke test: fire one pull, then shut down cleanly."""
    with Trigger() as trigger:
        print("Trigger armed. Firing one pull…")
        trigger.fire()
        # Give the worker time to complete the (blocking) pull before stop()
        # joins the thread: two 45° moves at 150°/s plus the dwell ~ 0.9 s.
        time.sleep(1.0)
        print("Done.")
    GPIO.cleanup()  # the standalone script owns the GPIO subsystem


if __name__ == "__main__":
    main()
