#!/usr/bin/env python3
"""
turret-ui — laptop app for the nerf turret.

Serves the control UI on http://localhost:8080 and relays everything to
turretd on the Pi:

    browser ──/ws, /api/*, /video.mjpg──► turret-ui ──► turretd (Pi, :8090)

Heartbeats come from the browser and are only forwarded, never generated
here: if the browser tab dies, the Pi's watchdog stops the turret.
It also runs the YOLO flying-object model on the laptop GPU and serves it on
:8100/detect (all interfaces, for the Pi only); on connecting it tells turretd
that URL. The Pi sends its timestamped frames there and fuses the boxes with
its own tracking (src/turretd/detect.py).

    browser ◄──► turret-ui ◄──► turretd ──frames──► :8100/detect (YOLO, here)

    python3 ui/turret_ui.py                       # Pi at clemenspi.local:8090
    python3 ui/turret_ui.py --pi 10.0.0.2         # direct link with fixed IPs
    python3 ui/turret_ui.py --pi localhost:8090   # against `python3 -m turretd --sim`
    python3 ui/turret_ui.py --no-detect           # without the YOLO model
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from aiohttp import WSMsgType, web

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
DEFAULT_PI = os.environ.get('TURRET_PI', 'clemenspi.local:8090')
RECONNECT_S = 2.0

PI = web.AppKey('pi', str)
SESSION = web.AppKey('session', aiohttp.ClientSession)
YOLO_KEY = web.AppKey('yolo', object)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO, 'Real-Time-Flying-Object-Detection_with_YOLOv8',
                               'weights', 'refined_3class', 'best.pt')


# ── YOLO on the laptop ───────────────────────────────────────────────────────
class Yolo:
    """Loads the model in the background (the UI is usable meanwhile) and runs
    inference on one worker thread (Apple GPU via MPS when available)."""

    def __init__(self, weights: str, imgsz: int, port: int) -> None:
        self.weights, self.imgsz, self.port = weights, imgsz, port
        self.model = None
        self.device = None
        self.error = None
        self.names = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            sys.path.insert(0, os.path.join(REPO, 'src'))
            import torch
            from detection import LocalYoloDetector
            self.device = 'mps' if torch.backends.mps.is_available() else (
                0 if torch.cuda.is_available() else 'cpu')
            det = LocalYoloDetector(self.weights, imgsz=self.imgsz, device=self.device)
            import numpy as np
            det.detect(np.zeros((self.imgsz * 9 // 16, self.imgsz, 3), np.uint8))   # warm up
            self.names, self.model = det.names, det
            print(f'YOLO ready on {self.device}: {list(self.names.values())}  '
                  f'(serving :{self.port}/detect)')
        except Exception as e:  # noqa: BLE001 — UI must keep working without it
            self.error = f'{type(e).__name__}: {e}'
            print(f'YOLO not available: {self.error}')

    def _detect(self, jpeg: bytes, conf: float) -> tuple[list, float]:
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('bad JPEG')
        t = time.perf_counter()
        self.model.conf = conf
        dets = self.model.detect(frame)
        return [d.to_dict() for d in dets], (time.perf_counter() - t) * 1000

    async def handle(self, request: web.Request) -> web.Response:
        if self.model is None:
            return web.json_response({'error': self.error or 'model loading'}, status=503)
        try:
            conf = float(request.query.get('conf', 0.4))
        except ValueError:
            conf = 0.4
        body = await request.read()
        try:
            dets, ms = await asyncio.get_running_loop().run_in_executor(
                self.pool, self._detect, body, conf)
        except ValueError as e:
            return web.json_response({'error': str(e)}, status=400)
        return web.json_response({'detections': dets, 'infer_ms': round(ms, 1)})

    def status(self) -> dict:
        return {'ready': self.model is not None, 'device': str(self.device),
                'error': self.error, 'port': self.port}


def create_detect_app(yolo: Yolo) -> web.Application:
    app = web.Application(client_max_size=8 * 1024 * 1024)
    app.router.add_post('/detect', yolo.handle)
    return app


def pi_url(app: web.Application, path: str, scheme: str = 'http') -> str:
    return f'{scheme}://{app[PI]}{path}'


# ── static ───────────────────────────────────────────────────────────────────
async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(os.path.join(STATIC, 'index.html'),
                            headers={'Cache-Control': 'no-cache'})


async def ui_info(request: web.Request) -> web.Response:
    yolo = request.app.get(YOLO_KEY)
    return web.json_response({'pi': request.app[PI], 'yolo': yolo.status() if yolo else None})


# ── REST proxy ───────────────────────────────────────────────────────────────
async def api_proxy(request: web.Request) -> web.Response:
    url = pi_url(request.app, '/api/' + request.match_info['tail'])
    body = await request.read()
    try:
        async with request.app[SESSION].request(
                request.method, url, data=body or None,
                headers={'Content-Type': request.headers.get('Content-Type', 'application/json')},
                timeout=aiohttp.ClientTimeout(total=5)) as r:
            return web.Response(body=await r.read(), status=r.status,
                                content_type=r.content_type)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response({'error': f'Pi not reachable at {request.app[PI]}: {e}'},
                                 status=502)


# ── video proxy ──────────────────────────────────────────────────────────────
async def video_proxy(request: web.Request) -> web.StreamResponse:
    try:
        upstream = await request.app[SESSION].get(
            pi_url(request.app, '/video.mjpg'),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=3, sock_read=10))
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return web.json_response({'error': str(e)}, status=502)
    resp = web.StreamResponse(headers={
        'Content-Type': upstream.headers.get('Content-Type', 'multipart/x-mixed-replace'),
        'Cache-Control': 'no-cache, private',
    })
    try:
        await resp.prepare(request)
        async for chunk in upstream.content.iter_any():
            await resp.write(chunk)
    except (ConnectionResetError, aiohttp.ClientError, asyncio.TimeoutError,
            asyncio.CancelledError):
        pass
    finally:
        upstream.release()
    return resp


# ── WebSocket relay ──────────────────────────────────────────────────────────
async def ws_relay(request: web.Request) -> web.WebSocketResponse:
    browser = web.WebSocketResponse(heartbeat=5.0)
    await browser.prepare(request)
    app = request.app
    state = {'pi': None}

    async def send_browser(msg: dict) -> None:
        if not browser.closed:
            await browser.send_json(msg)

    async def upstream_loop() -> None:
        """Keep a connection to the Pi; pump Pi → browser."""
        while not browser.closed:
            try:
                async with app[SESSION].ws_connect(pi_url(app, '/ws', 'ws'), heartbeat=5.0,
                                                   timeout=aiohttp.ClientWSTimeout(ws_close=3)
                                                   ) as pi:
                    state['pi'] = pi
                    await send_browser({'type': 'link', 'pi': True, 'host': app[PI]})
                    yolo = app.get(YOLO_KEY)
                    sock = pi.get_extra_info('sockname')
                    if yolo is not None and sock:
                        # our address as the Pi sees it
                        host = f'[{sock[0]}]' if ':' in sock[0] else sock[0]
                        await pi.send_str(json.dumps({'type': 'detector_url',
                                                      'url': f'http://{host}:{yolo.port}/detect'}))
                    async for msg in pi:
                        if msg.type == WSMsgType.TEXT and not browser.closed:
                            await browser.send_str(msg.data)
                        elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                await send_browser({'type': 'link', 'pi': False, 'host': app[PI],
                                    'error': str(e) or type(e).__name__})
            else:
                await send_browser({'type': 'link', 'pi': False, 'host': app[PI],
                                    'error': 'connection closed'})
            finally:
                state['pi'] = None
            await asyncio.sleep(RECONNECT_S)

    pump = asyncio.create_task(upstream_loop())
    try:
        async for msg in browser:
            if msg.type != WSMsgType.TEXT:
                continue
            pi = state['pi']
            if pi is not None and not pi.closed:
                await pi.send_str(msg.data)
                continue
            # Pi not connected: answer commands that expect a reply.
            try:
                cmd = json.loads(msg.data)
            except ValueError:
                continue
            if isinstance(cmd, dict) and cmd.get('id') is not None:
                await send_browser({'type': 'error', 'id': cmd['id'], 'cmd': cmd.get('type'),
                                    'msg': 'Pi not connected'})
    finally:
        pump.cancel()
        pi = state['pi']
        if pi is not None:
            await pi.close()   # the Pi's watchdog takes it from here
    return browser


# ── app ──────────────────────────────────────────────────────────────────────
async def _no_cache_static(request: web.Request, response: web.StreamResponse) -> None:
    # Revalidate app.js/style.css on every load (cheap 304 if unchanged), so a
    # new index.html never runs against a stale cached script.
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-cache'


def create_app(pi: str, yolo: 'Yolo | None' = None) -> web.Application:
    app = web.Application()
    app[PI] = pi
    if yolo is not None:
        app[YOLO_KEY] = yolo
    app.on_response_prepare.append(_no_cache_static)

    async def on_startup(app):
        app[SESSION] = aiohttp.ClientSession()

    async def on_cleanup(app):
        await app[SESSION].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get('/', index)
    app.router.add_get('/ui/info', ui_info)
    app.router.add_route('*', '/api/{tail:.*}', api_proxy)
    app.router.add_get('/video.mjpg', video_proxy)
    app.router.add_get('/ws', ws_relay)
    app.router.add_static('/static/', STATIC)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pi', default=DEFAULT_PI, help='turretd host[:port] (default %(default)s)')
    ap.add_argument('--host', default='127.0.0.1', help='bind address for the UI')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--no-detect', action='store_true', help='do not load the YOLO model')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS)
    ap.add_argument('--imgsz', type=int, default=640, help='YOLO inference size')
    ap.add_argument('--detect-port', type=int, default=8100)
    args = ap.parse_args()
    pi = args.pi if ':' in args.pi else f'{args.pi}:8090'
    yolo = None if args.no_detect else Yolo(args.weights, args.imgsz, args.detect_port)

    url = f'http://{"localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host}:{args.port}'
    print(f'turret-ui on {url}  →  turretd at {pi}')
    if not args.no_browser:
        asyncio.get_event_loop_policy()
        webbrowser.open(url)
    asyncio.run(_serve(pi, yolo, args))


async def _serve(pi: str, yolo, args) -> None:
    runners = []
    try:
        ui = web.AppRunner(create_app(pi, yolo))
        await ui.setup()
        runners.append(ui)
        await web.TCPSite(ui, args.host, args.port).start()
        if yolo is not None:
            det = web.AppRunner(create_detect_app(yolo), access_log=None)
            await det.setup()
            runners.append(det)
            await web.TCPSite(det, None, args.detect_port).start()   # all interfaces
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        for r in runners:
            await r.cleanup()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
