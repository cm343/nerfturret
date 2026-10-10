"""Detection → follow, against the simulated camera and a stand-in for the
laptop's YOLO server that reports the sim's red/white target as a helicopter."""

import asyncio
import time

import cv2
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from test_turretd_aim import TRUE_HFOV, TRUE_VFOV, CamRig
from test_turretd_controller import FAST, run
from turretd.camera import SimScene
from turretd.config import ConfigStore


@pytest.fixture
def cfg(tmp_path):
    c = ConfigStore(str(tmp_path))
    c.set_many({**FAST, 'camera.resolution': '640x480', 'aim.settle': 40,
                'safety.watchdog_ms': 3000, 'camera.hfov': TRUE_HFOV, 'camera.vfov': TRUE_VFOV,
                'aim.target_distance': 0.5, 'aim.dart_speed': 100, 'aim.follow_latency': 0,
                'detect.enabled': True, 'detect.max_rate': 10})
    return c


async def fake_yolo(request):
    img = cv2.imdecode(np.frombuffer(await request.read(), np.uint8), cv2.IMREAD_COLOR)
    red = (img[:, :, 2] > 170) & (img[:, :, 1] < 110) & (img[:, :, 0] < 110)
    ys, xs = np.nonzero(red)
    dets = []
    if len(xs) > 20:
        dets.append({'cls_id': 2, 'label': 'Helicopter', 'conf': 0.9,
                     'bbox': [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]})
    return web.json_response({'detections': dets, 'infer_ms': 5})


class DetRig(CamRig):
    async def __aenter__(self):
        app = web.Application()
        app.router.add_post('/detect', fake_yolo)
        self.server = TestServer(app)
        await self.server.start_server()
        await super().__aenter__()
        await self.cmd('detector_url', url=str(self.server.make_url('/detect')))
        return self

    async def __aexit__(self, *exc):
        await super().__aexit__(*exc)
        await self.server.close()

    def target_err(self):
        az, el = SimScene.target_at(time.monotonic() - self.cam.sim_t0)
        return max(abs(self.ctrl.position_deg('pan') - az), abs(self.ctrl.position_deg('tilt') - el))


async def first_result(r):
    await r.until(lambda: r.ctrl.detect.latest() is not None
                  and any(d['target'] for d in r.ctrl.detect.latest()['dets']), timeout=5)
    return r.ctrl.detect.latest()


def test_detections_reach_telemetry_and_classes_filter(cfg):
    async def body():
        async with DetRig(cfg) as r:
            res = await first_result(r)
            box = r.t()['detect']['result']['boxes'][0]
            assert box['label'] == 'Helicopter' and box['target']
            r.ctrl.set_config({'detect.helicopter': False})
            await r.until(lambda: r.ctrl.detect.latest()['id'] > res['id']
                          and not r.ctrl.detect.latest()['dets'][0]['target'])
    run(body())


def test_clicking_a_box_follows_the_object(cfg):
    async def body():
        async with DetRig(cfg) as r:
            res = await first_result(r)
            await r.cmd('follow_box', rid=res['id'], i=0)
            errs = []
            for _ in range(30):
                await asyncio.sleep(0.1)
                assert r.ctrl.aim.status['state'] == 'following', r.ctrl.aim.status
                errs.append(r.target_err())
            assert sum(errs[10:]) / len(errs[10:]) < 0.8, errs
    run(body())


def test_auto_track_engages_and_stop_switches_it_off(cfg):
    cfg.set_many({'detect.auto_track': True})
    async def body():
        async with DetRig(cfg) as r:
            await r.until(lambda: r.ctrl.aim.following, timeout=5)
            assert any('auto-track: following Helicopter' in e['msg'] for e in r.events)
            await r.cmd('aim_cancel')                              # Stop follow
            assert not r.ctrl.cfg['detect.auto_track'] and not r.ctrl.aim.following
            await asyncio.sleep(0.5)
            assert not r.ctrl.aim.following                        # stays off
    run(body())


def test_manual_move_pauses_auto_track(cfg):
    cfg.set_many({'detect.auto_track': True, 'detect.manual_pause': 30})
    async def body():
        async with DetRig(cfg) as r:
            await r.cmd('jog', axis='pan', dir=1, deg=1)
            await asyncio.sleep(1.0)
            assert not r.ctrl.aim.following
    run(body())
