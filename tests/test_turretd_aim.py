"""Closed-loop aiming against the simulated camera: a pinhole view of a textured
world that follows the simulated pan/tilt. The scene uses the TRUE field of
view; the controller is configured with a wrong one, like an uncalibrated rig."""

import asyncio
import math

import pytest

from test_turretd_controller import FAST, Rig, run
from turretd.camera import CameraService
from turretd.config import ConfigStore

TRUE_HFOV, TRUE_VFOV = 53.5, 30.1


@pytest.fixture
def cfg(tmp_path):
    c = ConfigStore(str(tmp_path))
    c.set_many({**FAST, 'camera.resolution': '640x480', 'aim.settle': 40,
                'safety.watchdog_ms': 3000,
                'camera.hfov': 60.0, 'camera.vfov': 34.0})       # deliberately wrong
    return c


class CamRig(Rig):
    def __init__(self, cfg):
        cam = CameraService('sim')
        super().__init__(cfg, cam)
        cam.view = lambda: (self.ctrl.position_deg('pan'), self.ctrl.position_deg('tilt'))
        self.cam = cam

    async def __aenter__(self):
        await super().__aenter__()
        self.cam.start({'camera.resolution': '640x480', 'camera.fps': 30,
                        'camera.hfov': TRUE_HFOV, 'camera.vfov': TRUE_VFOV})
        await asyncio.get_running_loop().run_in_executor(None, self.cam.gray_after, 0, 2.0)
        return self

    async def __aexit__(self, *exc):
        self.cam.stop()
        await super().__aexit__(*exc)

    async def settled(self):
        """Turret at rest and a frame exposed after the settle time exists."""
        await asyncio.sleep(0.05)
        await self.until(lambda: self.ctrl.aim.still_since is not None and self.cam.gray_ts
                         >= self.ctrl.aim.still_since + self.ctrl.cfg['aim.settle'] / 1000 + 0.05)


def true_angle(frac, fov):
    return math.degrees(math.atan(frac * math.tan(math.radians(fov) / 2)))


async def wait_aim(r, timeout=10.0):
    await r.until(lambda: r.ctrl.aim.status['busy'] is None, timeout)
    return r.ctrl.aim.status


def test_auto_aim_corrects_a_wrong_fov(cfg):
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            res = await r.cmd('aim_offset', x=0.55, y=0.4)
            assert res == {'mode': 'auto'}
            st = await wait_aim(r)
            assert st['state'] == 'locked', (st, [e['msg'] for e in r.events])
            assert any('tracking locked (spot + scene' in e['msg'] for e in r.events)
            pan, tilt = r.ctrl.position_deg('pan'), r.ctrl.position_deg('tilt')
            assert pan == pytest.approx(true_angle(0.55, TRUE_HFOV), abs=0.15)
            assert tilt == pytest.approx(-true_angle(0.4, TRUE_VFOV), abs=0.15)
    run(body())


def test_open_mode_keeps_the_fov_error(cfg):
    cfg.set_many({'aim.mode': 'open'})
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            assert await r.cmd('aim_offset', x=0.55, y=0) == {'mode': 'open'}
            await r.until(lambda: not r.motion.axis(0).moving and r.motion.axis(0).pos != 0)
            err = r.ctrl.position_deg('pan') - true_angle(0.55, TRUE_HFOV)
            assert abs(err) > 1.0                       # 60° vs 53.5°: clearly off
    run(body())


def test_plain_spot_is_still_refined_from_scene_features(cfg):
    cfg.set_many({'aim.min_texture': 50})                     # the spot itself never matches
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            assert await r.cmd('aim_offset', x=0.55, y=0.4) == {'mode': 'auto'}
            st = await wait_aim(r)
            assert st['state'] == 'locked' and '(scene, ' in st['msg'], st
            assert r.ctrl.position_deg('pan') == pytest.approx(true_angle(0.55, TRUE_HFOV), abs=0.15)
            assert r.ctrl.position_deg('tilt') == pytest.approx(-true_angle(0.4, TRUE_VFOV), abs=0.15)
    run(body())


def test_unmeasurable_move_keeps_open_loop_and_says_why(cfg):
    cfg.set_many({'aim.min_texture': 50, 'aim.min_points': 200})   # nothing is good enough
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            await r.cmd('aim_offset', x=0.3, y=0)
            st = await wait_aim(r)
            assert st['state'] == 'lost'
            assert any(e['level'] == 'warn' and 'tracking failed after the move' in e['msg']
                       for e in r.events)
    run(body())


def test_click_while_moving_is_open_loop(cfg):
    async def body():
        async with CamRig(cfg) as r:
            await r.cmd('move', axis='pan', dir=1)
            await asyncio.sleep(0.1)
            assert await r.cmd('aim_offset', x=0.3, y=0) == {'mode': 'open'}
            assert 'moving' in r.ctrl.aim.status['msg']
    run(body())


def test_manual_move_cancels_refinement(cfg):
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            await r.cmd('aim_offset', x=0.55, y=0.4)
            await r.cmd('jog', axis='pan', dir=1, deg=1)
            assert r.ctrl.aim.status['msg'] == 'cancelled'
    run(body())


@pytest.mark.parametrize('axis, true_fov', [('pan', TRUE_HFOV), ('tilt', TRUE_VFOV)])
def test_fov_calibration_recovers_the_true_fov(cfg, axis, true_fov):
    cfg.set_many({'camera.hfov': 90.0, 'camera.vfov': 10.0})      # badly off, like the real rig
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            await r.cmd('fov_calibrate', axis=axis)
            st = await wait_aim(r, timeout=30)
            assert st['state'] == 'done', st
            key = 'camera.hfov' if axis == 'pan' else 'camera.vfov'
            assert r.ctrl.cfg[key] == pytest.approx(true_fov, abs=0.1)
            await r.until(lambda: not r.motion.axis(r.ctrl.axis_index(axis)).moving)
            assert r.ctrl.position_deg(axis) == pytest.approx(0, abs=0.05)   # returned
    run(body())


def test_failed_calibration_says_why_each_sweep_stopped(cfg):
    cfg.set_many({'aim.min_points': 200})                     # no step ever has enough
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            await r.cmd('fov_calibrate', axis='pan')
            st = await wait_aim(r, timeout=30)
            assert st['state'] == 'failed', (st, [e['msg'] for e in r.events])
            assert 'usable steps' in st['msg']                 # fewer than 3
            assert 'features' in st['msg']                     # …and why each sweep ended
            await r.until(lambda: not r.motion.axis(0).moving)
            assert r.ctrl.position_deg('pan') == pytest.approx(0, abs=0.05)
    run(body())


# ── follow mode ──────────────────────────────────────────────────────────────
def _target_uv(r):
    """Where the sim's moving target is in the latest frame (normalized)."""
    from turretd.camera import SimScene
    az, el = SimScene.target_at(r.cam.gray_ts - r.cam.sim_t0)
    pan, tilt = r.ctrl.position_deg('pan'), r.ctrl.position_deg('tilt')
    k_h = math.tan(math.radians(TRUE_HFOV) / 2)
    k_v = math.tan(math.radians(TRUE_VFOV) / 2)
    return math.tan(math.radians(az - pan)) / k_h, math.tan(math.radians(tilt - el)) / k_v


def test_follow_keeps_the_turret_on_a_moving_target(cfg):
    from turretd.camera import SimScene
    import time as _t
    cfg.set_many({'aim.mode': 'follow', 'camera.hfov': TRUE_HFOV, 'camera.vfov': TRUE_VFOV,
                  'aim.click_latency': 0, 'aim.target_distance': 0.5, 'aim.dart_speed': 100,
                  'aim.follow_latency': 0})                       # ~no lead: aim at the target
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            u, v = _target_uv(r)
            assert await r.cmd('aim_offset', x=u, y=v) == {'mode': 'follow'}
            errs, green = [], False
            for _ in range(40):                                   # 4 s
                await asyncio.sleep(0.1)
                st = r.ctrl.aim.status
                assert st['state'] == 'following', st
                green |= bool(st.get('on_target'))
                az, el = SimScene.target_at(_t.monotonic() - r.cam.sim_t0)
                errs.append(max(abs(r.ctrl.position_deg('pan') - az),
                                abs(r.ctrl.position_deg('tilt') - el)))
            tail = errs[10:]                                      # after 1 s: on it
            assert sum(tail) / len(tail) < 0.6 and max(tail) < 1.5, errs
            assert green
            await r.cmd('aim_cancel')                             # Stop button
            assert not r.ctrl.aim.following
            assert r.ctrl.aim.status['msg'] == 'follow stopped'
    run(body())


def test_follow_needs_something_to_lock_on(cfg):
    cfg.set_many({'aim.mode': 'follow', 'aim.min_texture': 50})
    async def body():
        async with CamRig(cfg) as r:
            await r.settled()
            await r.cmd('aim_offset', x=0.1, y=0.1)
            await r.until(lambda: r.ctrl.aim.status['state'] == 'lost')
            assert 'texture' in r.ctrl.aim.status['msg']
    run(body())
