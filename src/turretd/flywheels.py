"""
The two flywheel ESCs (left GPIO 12, right GPIO 13) as one unit for turretd.

Wraps esc.ESC. The stop pulse goes out from setup() on, so the ESCs can arm
whenever their power comes on. The first arm() additionally waits arm_time with
the stop pulse before throttle is allowed, in case they were powered just now.
Ramping is driven by the controller's tick (ramp_tick), never by blocking calls.
"""

import asyncio
import os
import subprocess

from esc import ESC, HardwarePwmOutput, LgpioServoOutput

ESC_PINS = (12, 13)        # BCM: left, right


class SimServoOutput:
    """No hardware: remembers the last pulse width per pin."""

    def __init__(self) -> None:
        self.pulse: dict[int, int] = {}

    def set_pulse_us(self, pin: int, us: int) -> None:
        self.pulse[pin] = int(us)

    def close(self) -> None:
        pass


def _set_pwm_pinmux() -> None:
    """Route GPIO 12/13 to the PWM block (ALT0). lgpio or RPi.GPIO may have left
    them as plain outputs, which silently disconnects the hardware PWM."""
    try:
        subprocess.run(['pinctrl', 'set', ','.join(map(str, ESC_PINS)), 'a0'],
                       check=True, capture_output=True, timeout=2)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f'could not switch GPIO {ESC_PINS} to PWM mode: {e}') from e


def make_output(backend: str):
    if backend == 'sim':
        return SimServoOutput()
    if backend == 'hwpwm':
        out = HardwarePwmOutput()
        _set_pwm_pinmux()
        return out
    return LgpioServoOutput()


def kill_hwpwm(root: str = '/sys/class/pwm/pwmchip0') -> None:
    """Best effort, no Python state needed: switch both PWM channels off so the
    ESCs see no signal and stop. Used when turretd's main process dies, because
    hardware PWM would otherwise keep sending the last throttle."""
    for ch in sorted(set(HardwarePwmOutput.CHANNELS[p] for p in ESC_PINS)):
        try:
            with open(os.path.join(root, f'pwm{ch}', 'enable'), 'w') as f:
                f.write('0')
        except OSError:
            pass


class Flywheels:
    def __init__(self, output, *, min_us: int = 1000, max_us: int = 2000) -> None:
        self.output = output
        self.escs = [ESC(output, pin, min_us=min_us, max_us=max_us) for pin in ESC_PINS]
        self.armed = False

    def setup(self) -> None:
        for e in self.escs:
            e.setup()

    async def arm(self, seconds: float) -> None:
        """Hold the stop pulse for `seconds` (first time only), then allow throttle."""
        for e in self.escs:
            e.stop()
        if not self.armed and seconds > 0:
            await asyncio.sleep(seconds)
        for e in self.escs:
            e.armed = True
        self.armed = True

    @property
    def throttles(self) -> tuple[float, float]:
        return tuple(e.throttle for e in self.escs)

    @property
    def pulses_us(self) -> tuple[int, int]:
        return tuple(e.pulse_us(e.throttle) for e in self.escs)

    def ramp_tick(self, targets: tuple[float, float], max_delta: float) -> bool:
        """One ramp step toward (left, right). Returns True when both are there."""
        if not self.armed:
            return False
        done = True
        for e, t in zip(self.escs, targets):
            e.max_throttle = 1.0          # the controller already applied the cap
            done = e.ramp_step(t, max_delta) and done
        return done

    def stop(self) -> None:
        """Cut both to the stop pulse immediately; the wheels coast down."""
        for e in self.escs:
            if e._ready:
                e.stop()

    def close(self) -> None:
        for e in self.escs:
            e.cleanup()
        self.output.close()
