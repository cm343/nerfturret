"""Tests for turretd's controller and API, on simulated hardware.

The motion worker runs as a thread with sim pins, the ESCs on a SimServoOutput,
and the config is tuned for speed (no arming wait, short stroke, fast loader),
so whole fire cycles finish in well under a second.
"""

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from turretd.config import ConfigError, ConfigStore
from turretd.controller import AXES, AXIS_PINS, CommandError, Controller
from turretd.flywheels import Flywheels, SimServoOutput
from turretd.motion import MotionClient
from turretd.server import create_app

FAST = {
    'flywheel.arm_time': 0, 'flywheel.spinup_time': 0.05, 'flywheel.ramp_rate': 1000,
    'loader.stroke': 30, 'loader.push_speed': 3600, 'loader.return_speed': 3600,
    'loader.accel': 50000, 'loader.front_dwell': 0, 'loader.rate': 10,
    'pan.speed': 720, 'pan.accel': 5000, 'tilt.speed': 720, 'tilt.accel': 5000,
}


@pytest.fixture
def cfg(tmp_path):
    c = ConfigStore(str(tmp_path))
    c.set_many(FAST)
    return c


class Rig:
    def __init__(self, cfg, camera=None):
        self.motion = MotionClient([AXIS_PINS[a] for a in AXES], 'sim', use_process=False)
        self.out = SimServoOutput()
        self.fly = Flywheels(self.out)
        self.ctrl = Controller(cfg, self.motion, self.fly, camera)
        self.events = []
        self.ctrl.listeners.append(lambda m: m.get('type') == 'event' and self.events.append(m))

    async def __aenter__(self):
        self.motion.start()
        await self.ctrl.start()
        self._hb = asyncio.create_task(self._heartbeat())
        return self

    async def _heartbeat(self):
        while True:
            await self.ctrl.handle({'type': 'heartbeat'})
            await asyncio.sleep(0.05)

    async def __aexit__(self, *exc):
        self._hb.cancel()
        await self.ctrl.shutdown()
        self.motion.stop()

    async def cmd(self, type_, **kw):
        return await self.ctrl.handle({'type': type_, **kw})

    async def until(self, pred, timeout=5.0):
        end = asyncio.get_running_loop().time() + timeout
        while not pred():
            if asyncio.get_running_loop().time() > end:
                raise AssertionError('condition not reached')
            await asyncio.sleep(0.01)

    def t(self):
        return self.ctrl.telemetry()


def run(coro):
    return asyncio.run(coro)


# ── config store ─────────────────────────────────────────────────────────────
def test_config_defaults_validation_and_persistence(tmp_path):
    c = ConfigStore(str(tmp_path))
    assert c['flywheel.throttle'] == 30.0
    applied = c.set_many({'flywheel.throttle': 250, 'fire.mode': 'burst', 'pan.invert': 'true'})
    assert applied == {'flywheel.throttle': 100.0, 'fire.mode': 'burst', 'pan.invert': True}
    with pytest.raises(ConfigError):
        c.set_many({'fire.mode': 'laser'})
    with pytest.raises(ConfigError):
        c.set_many({'nope.key': 1})
    saved = json.loads((tmp_path / 'turret.json').read_text())
    assert saved == {'flywheel.throttle': 100.0, 'fire.mode': 'burst', 'pan.invert': True}
    assert ConfigStore(str(tmp_path))['fire.mode'] == 'burst'


def test_config_bad_file_entries_are_reported_not_fatal(tmp_path):
    (tmp_path / 'turret.json').write_text(json.dumps({'fire.mode': 'laser', 'x.y': 1,
                                                       'pan.speed': 30}))
    c = ConfigStore(str(tmp_path))
    assert c['pan.speed'] == 30 and c['fire.mode'] == 'single'
    assert len(c.load_errors) == 2


def test_presets_round_trip(tmp_path):
    c = ConfigStore(str(tmp_path))
    c.set_many({'flywheel.throttle': 55})
    c.save_preset('indoor')
    c.set_many({'flywheel.throttle': 20})
    assert c.list_presets() == ['indoor']
    assert c.load_preset('indoor') == {'flywheel.throttle': 55.0}
    with pytest.raises(ConfigError):
        c.save_preset('../evil')
    c.delete_preset('indoor')
    assert c.list_presets() == []


# ── controller ───────────────────────────────────────────────────────────────
def test_starts_safe(cfg):
    async def body():
        async with Rig(cfg) as r:
            t = r.t()
            assert t['state'] == 'DISARMED'
            assert not any(a['enabled'] for a in t['axes'].values())
            assert r.out.pulse == {12: 1000, 13: 1000}       # stop pulse from the start
    run(body())


def test_spinning_needs_arming(cfg):
    async def body():
        async with Rig(cfg) as r:
            with pytest.raises(CommandError, match='arm first'):
                await r.cmd('spin', on=True)
    run(body())


def test_disarmed_fire_loads_without_flywheels(cfg):
    async def body():
        async with Rig(cfg) as r:
            assert r.t()['fire']['action'] == 'load'
            await r.cmd('fire_start', mode='burst')
            await r.until(lambda: not r.ctrl.firing)
            assert r.ctrl.shots == 3 and r.motion.axis(2).pos == 0
            assert r.out.pulse == {12: 1000, 13: 1000}       # flywheels never ran
            assert not r.ctrl.spin_on
            assert any('loaded 3 (flywheels off)' in e['msg'] for e in r.events)
    run(body())


def test_disarmed_loading_can_be_switched_off(cfg):
    cfg.set_many({'fire.load_when_disarmed': False})
    async def body():
        async with Rig(cfg) as r:
            assert r.t()['fire']['action'] is None
            with pytest.raises(CommandError, match='arm first'):
                await r.cmd('fire_start')
    run(body())


def test_armed_without_auto_spinup_loads_with_wheels_off(cfg):
    cfg.set_many({'fire.auto_spinup': False})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            assert r.t()['fire']['action'] == 'load'
            await r.cmd('fire_start')
            await r.until(lambda: not r.ctrl.firing)
            assert r.ctrl.shots == 1 and not r.ctrl.spin_on
            await r.cmd('spin', on=True)                     # wheels on → real firing
            assert r.t()['fire']['action'] == 'fire'
    run(body())


def test_arm_spin_and_trim(cfg):
    cfg.set_many({'flywheel.throttle': 40, 'flywheel.trim': 10})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            await r.cmd('spin', on=True)
            await r.until(lambda: r.ctrl.spun_up)
            assert r.t()['flywheels']['left'] == 45.0 and r.t()['flywheels']['right'] == 35.0
            assert r.out.pulse == {12: 1450, 13: 1350}
            await r.cmd('disarm')
            await asyncio.sleep(0.05)
            assert r.out.pulse == {12: 1000, 13: 1000}
    run(body())


def test_throttle_cap_applies_after_trim(cfg):
    cfg.set_many({'flywheel.throttle': 58, 'flywheel.trim': 10, 'flywheel.max_throttle': 60})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            await r.cmd('spin', on=True)
            await r.until(lambda: r.ctrl.spun_up)
            assert r.t()['flywheels']['left'] == 60.0
    run(body())


@pytest.mark.parametrize('mode, expected', [('single', 1), ('burst', 3)])
def test_fire_modes_cycle_the_loader(cfg, mode, expected):
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            await r.cmd('fire_start', mode=mode)            # auto spin-up
            await r.until(lambda: not r.ctrl.firing)
            assert r.ctrl.shots == expected
            assert r.ctrl.ammo == 12 - expected
            assert r.motion.axis(2).pos == 0                # loader back at rest
            pulses = r.motion.worker.pins.pulses[AXIS_PINS['loader'].step]
            assert pulses == expected * 2 * r.ctrl.to_steps('loader', 30)
    run(body())


def test_auto_fire_stops_on_request_and_on_empty_magazine(cfg):
    cfg.set_many({'loader.magazine': 4})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            await r.cmd('fire_start', mode='auto')
            await r.until(lambda: not r.ctrl.firing, timeout=10)
            assert r.ctrl.ammo == 0 and r.ctrl.shots == 4
            assert any('magazine empty' in e['msg'] for e in r.events)
            await r.cmd('reload')
            await r.cmd('fire_start', mode='auto')
            await r.until(lambda: r.ctrl.shots >= 5)
            await r.cmd('fire_stop')
            await r.until(lambda: not r.ctrl.firing)
            assert 0 < r.ctrl.ammo < 4
    run(body())


def test_estop_halts_and_latches(cfg):
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            await r.cmd('spin', on=True)
            await r.cmd('move', axis='pan', dir=1)
            await asyncio.sleep(0.1)
            await r.cmd('estop')
            t = r.t()
            assert t['state'] == 'ESTOP' and not t['flywheels']['on']
            assert r.out.pulse == {12: 1000, 13: 1000}
            await asyncio.sleep(0.05)
            assert not r.motion.axis(0).moving
            with pytest.raises(CommandError, match='E-STOP'):
                await r.cmd('arm')
            with pytest.raises(CommandError, match='E-STOP'):
                await r.cmd('move', axis='pan', dir=1)
            await r.cmd('reset')
            assert r.t()['state'] == 'DISARMED'
    run(body())


def test_watchdog_estops_when_heartbeats_stop(cfg):
    cfg.set_many({'safety.watchdog_ms': 150})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('arm')
            await r.until(lambda: r.ctrl.state == 'ARMED')
            r._hb.cancel()
            await r.until(lambda: r.ctrl.estop_reason is not None, timeout=1)
            assert 'link lost' in r.ctrl.estop_reason
    run(body())


def test_watchdog_ignores_missing_heartbeat_when_idle(cfg):
    cfg.set_many({'safety.watchdog_ms': 100})
    async def body():
        async with Rig(cfg) as r:
            r._hb.cancel()
            await asyncio.sleep(0.3)
            assert r.ctrl.estop_reason is None
    run(body())


def test_reference_limits_and_degrees(cfg):
    cfg.set_many({'pan.gear_ratio': 3.0, 'pan.limit_min': -10, 'pan.limit_max': 20})
    async def body():
        async with Rig(cfg) as r:
            with pytest.raises(CommandError, match='not referenced'):
                await r.cmd('goto', axis='pan', deg=5)
            await r.cmd('set_zero', axis='pan')
            await r.cmd('goto', axis='pan', deg=90)              # clamped to +20
            await r.until(lambda: not r.motion.axis(0).moving and r.motion.axis(0).pos > 0)
            assert r.t()['axes']['pan']['pos'] == pytest.approx(20, abs=0.05)
            # 20° at the turret × gear 3 × 200·16/360 µsteps per motor degree
            assert r.motion.axis(0).pos == round(20 * 3 * 200 * 16 / 360)
            await r.cmd('move', axis='pan', dir=-1)
            await r.until(lambda: r.motion.axis(0).pos < 0 and not r.motion.axis(0).moving)
            assert r.t()['axes']['pan']['pos'] == pytest.approx(-10, abs=0.05)
    run(body())


def test_invert_flips_direction(cfg):
    cfg.set_many({'tilt.invert': True})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('set_zero', axis='tilt')
            await r.cmd('jog', axis='tilt', dir=1, deg=2)
            await r.until(lambda: r.motion.axis(1).pos != 0 and not r.motion.axis(1).moving)
            assert r.motion.axis(1).pos < 0
            assert r.t()['axes']['tilt']['pos'] == pytest.approx(2, abs=0.05)
    run(body())


def test_set_limit_from_current_position(cfg):
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('set_zero', axis='tilt')
            await r.cmd('jog', axis='tilt', dir=1, deg=12)
            await r.until(lambda: r.motion.axis(1).pos > 0 and not r.motion.axis(1).moving)
            await r.cmd('set_limit', axis='tilt', which='max')
            assert cfg['tilt.limit_max'] == pytest.approx(12, abs=0.05)
            with pytest.raises(CommandError, match='below'):
                await r.cmd('set_limit', axis='tilt', which='min')   # 12 is not below max
    run(body())


def test_disabling_drops_reference(cfg):
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('set_zero', axis='pan')
            await r.cmd('enable', axis='pan', on=False)
            assert not r.ctrl.referenced['pan']
            assert r.t()['axes']['pan']['limits'] is None
    run(body())


def test_loader_calibration(cfg):
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('loader_set_rest')
            await r.cmd('raw_jog', axis='loader', deg=200)
            await r.until(lambda: r.motion.axis(2).pos > 0 and not r.motion.axis(2).moving)
            await r.cmd('loader_set_stroke')
            assert cfg['loader.stroke'] == pytest.approx(200, abs=0.5)
    run(body())


def test_aim_offset_turns_by_fov_fraction(cfg):
    cfg.set_many({'camera.hfov': 60, 'camera.vfov': 40})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('aim_offset', x=0.5, y=-0.5)       # right and up
            await r.until(lambda: r.motion.axis(0).pos > 0 and not r.motion.axis(0).moving
                          and not r.motion.axis(1).moving)
            t = r.t()['axes']
            # pinhole: atan(0.5·tan 30°) = 16.10°, atan(0.5·tan 20°) = 10.31°
            assert t['pan']['pos'] == pytest.approx(16.10, abs=0.05)
            assert t['tilt']['pos'] == pytest.approx(10.31, abs=0.05)
    run(body())


def test_aim_offset_is_measured_from_the_crosshair(cfg):
    cfg.set_many({'camera.hfov': 60, 'camera.crosshair_x': 0.5})
    async def body():
        async with Rig(cfg) as r:
            await r.cmd('aim_offset', x=-0.5, y=0)         # clicked the image centre
            await r.until(lambda: r.motion.axis(0).pos != 0 and not r.motion.axis(0).moving)
            assert r.t()['axes']['pan']['pos'] == pytest.approx(-16.10, abs=0.05)
    run(body())


def test_unknown_command(cfg):
    async def body():
        async with Rig(cfg) as r:
            with pytest.raises(CommandError, match='unknown'):
                await r.cmd('self_destruct')
    run(body())


# ── HTTP / WebSocket API ─────────────────────────────────────────────────────
def test_api_end_to_end(cfg):
    async def body():
        rig = Rig(cfg)
        rig.motion.start()
        app = create_app(rig.ctrl)
        async with TestClient(TestServer(app)) as client:
            schema = await (await client.get('/api/schema')).json()
            keys = {p['key'] for p in schema['params']}
            assert {'flywheel.throttle', 'pan.limit_max', 'camera.crosshair_x'} <= keys

            r = await client.patch('/api/config', json={'flywheel.throttle': 42})
            assert (await r.json())['applied'] == {'flywheel.throttle': 42.0}
            r = await client.patch('/api/config', json={'fire.mode': 'laser'})
            assert r.status == 400

            assert (await client.post('/api/presets/test')).status == 200
            assert (await (await client.get('/api/presets')).json())['presets'] == ['test']

            ws = await client.ws_connect('/ws')
            hello = await ws.receive_json()
            assert hello['type'] == 'hello' and hello['config']['flywheel.throttle'] == 42.0
            await ws.send_json({'type': 'spin', 'on': True, 'id': 1})
            seen_error = seen_tele = False
            for _ in range(20):
                m = await ws.receive_json(timeout=2)
                if m['type'] == 'error' and m['id'] == 1:
                    assert 'arm first' in m['msg']
                    seen_error = True
                if m['type'] == 'telemetry':
                    seen_tele = True
                if seen_error and seen_tele:
                    break
            assert seen_error and seen_tele
            await ws.close()
        rig.motion.stop()
    run(body())
