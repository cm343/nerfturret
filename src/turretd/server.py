"""
HTTP + WebSocket API of turretd.

  GET    /api/schema                 parameter schema (builds the UI config page)
  GET    /api/config                 current values (+ problems found while loading)
  PATCH  /api/config                 {key: value, …} → validated, applied, saved
  GET    /api/presets                preset names
  POST   /api/presets/{name}         save current values as a preset
  POST   /api/presets/{name}/load    apply a preset
  DELETE /api/presets/{name}
  GET    /api/info                   pins, backends, version
  GET    /video.mjpg                 live camera (multipart MJPEG)
  GET    /snapshot.jpg               latest frame
  GET    /ws                         WebSocket

WebSocket protocol (JSON text frames):
  → {"type": "<command>", "id": 7, …}   see Controller._cmd_* (heartbeat every 100 ms)
  ← {"type": "ack", "id": 7, "result": …} | {"type": "error", "id": 7, "msg": "…"}
  ← {"type": "telemetry", …}             20 Hz
  ← {"type": "event", "level", "msg"}    as they happen
  ← {"type": "config", "values": {…}}    after any config change (all clients)
  ← {"type": "hello", …}                 once, on connect
"""

import asyncio
import json
import weakref

from aiohttp import WSMsgType, web

from .config import ConfigError, schema_json
from .controller import AXIS_PINS, CommandError, Controller
from .flywheels import ESC_PINS

VERSION = '0.1.0'
TELEMETRY_HZ = 20
BOUNDARY = 'frame'

CTRL = web.AppKey('controller', Controller)
CLIENTS = web.AppKey('clients', weakref.WeakSet)
TELEMETRY_TASK = web.AppKey('telemetry_task', asyncio.Task)


def _json(data, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _error(msg: str, status: int = 400) -> web.Response:
    return _json({'error': msg}, status)


# ── REST ─────────────────────────────────────────────────────────────────────
async def get_schema(request: web.Request) -> web.Response:
    return _json(schema_json())


async def get_config(request: web.Request) -> web.Response:
    cfg = request.app[CTRL].cfg
    return _json({'values': cfg.snapshot(), 'load_errors': cfg.load_errors})


async def patch_config(request: web.Request) -> web.Response:
    try:
        changes = await request.json()
        if not isinstance(changes, dict):
            raise ConfigError('body must be a JSON object')
        applied = request.app[CTRL].set_config(changes)
    except (ConfigError, json.JSONDecodeError) as e:
        return _error(str(e))
    return _json({'applied': applied})


async def list_presets(request: web.Request) -> web.Response:
    return _json({'presets': request.app[CTRL].cfg.list_presets()})


async def save_preset(request: web.Request) -> web.Response:
    ctrl = request.app[CTRL]
    try:
        ctrl.cfg.save_preset(request.match_info['name'])
    except ConfigError as e:
        return _error(str(e))
    ctrl.broadcast({'type': 'presets', 'presets': ctrl.cfg.list_presets()})
    return _json({'ok': True})


async def load_preset(request: web.Request) -> web.Response:
    try:
        changed = request.app[CTRL].load_preset(request.match_info['name'])
    except ConfigError as e:
        return _error(str(e), 404)
    return _json({'changed': changed})


async def delete_preset(request: web.Request) -> web.Response:
    ctrl = request.app[CTRL]
    try:
        ctrl.cfg.delete_preset(request.match_info['name'])
    except ConfigError as e:
        return _error(str(e), 404)
    ctrl.broadcast({'type': 'presets', 'presets': ctrl.cfg.list_presets()})
    return _json({'ok': True})


async def get_info(request: web.Request) -> web.Response:
    ctrl = request.app[CTRL]
    return _json({
        'version': VERSION,
        'pins': {a: {'step': p.step, 'dir': p.dir, 'en': p.en} for a, p in AXIS_PINS.items()},
        'esc_pins': {'left': ESC_PINS[0], 'right': ESC_PINS[1]},
        'motion_backend': ctrl.motion.backend,
        'camera_backend': ctrl.camera.backend if ctrl.camera else None,
        'config_path': ctrl.cfg.path,
    })


# ── video ────────────────────────────────────────────────────────────────────
async def video(request: web.Request) -> web.StreamResponse:
    cam = request.app[CTRL].camera
    if cam is None:
        return _error('no camera', 404)
    resp = web.StreamResponse(headers={
        'Content-Type': f'multipart/x-mixed-replace; boundary={BOUNDARY}',
        'Cache-Control': 'no-cache, private', 'Pragma': 'no-cache',
    })
    await resp.prepare(request)
    loop = asyncio.get_running_loop()
    seq = -1
    try:
        while True:
            seq, frame = await loop.run_in_executor(None, cam.wait_frame, seq, 1.0)
            if frame is None:
                continue
            await resp.write(
                f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n'
                f'Content-Length: {len(frame)}\r\n\r\n'.encode() + frame + b'\r\n')
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def snapshot(request: web.Request) -> web.Response:
    cam = request.app[CTRL].camera
    if cam is None or cam.frame is None:
        return _error('no frame yet', 404)
    return web.Response(body=cam.frame, content_type='image/jpeg')


# ── WebSocket ────────────────────────────────────────────────────────────────
async def websocket(request: web.Request) -> web.WebSocketResponse:
    ctrl = request.app[CTRL]
    ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=1 << 20)
    await ws.prepare(request)
    request.app[CLIENTS].add(ws)
    await ws.send_json({
        'type': 'hello', 'version': VERSION, 'config': ctrl.cfg.snapshot(),
        'presets': ctrl.cfg.list_presets(), 'load_errors': ctrl.cfg.load_errors,
    })
    await ws.send_json(ctrl.telemetry())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                cmd = json.loads(msg.data)
                if not isinstance(cmd, dict):
                    raise ValueError
            except ValueError:
                await ws.send_json({'type': 'error', 'msg': 'invalid JSON command'})
                continue
            cid = cmd.get('id')
            try:
                result = await ctrl.handle(cmd)
                if cid is not None:
                    await ws.send_json({'type': 'ack', 'id': cid, 'result': result})
            except (CommandError, ConfigError) as e:
                await ws.send_json({'type': 'error', 'id': cid, 'cmd': cmd.get('type'),
                                    'msg': str(e)})
    finally:
        request.app[CLIENTS].discard(ws)
    return ws


def _send_all(app: web.Application, message: dict) -> None:
    data = json.dumps(message)
    for ws in list(app[CLIENTS]):
        if not ws.closed:
            asyncio.ensure_future(_safe_send(ws, data))


async def _safe_send(ws, data: str) -> None:
    try:
        await ws.send_str(data)
    except (ConnectionResetError, RuntimeError):
        pass


async def _telemetry_loop(app: web.Application) -> None:
    ctrl = app[CTRL]
    while True:
        await asyncio.sleep(1 / TELEMETRY_HZ)
        if len(app[CLIENTS]):
            _send_all(app, ctrl.telemetry())


# ── app ──────────────────────────────────────────────────────────────────────
def create_app(ctrl: Controller, *, start_controller: bool = True) -> web.Application:
    app = web.Application()
    app[CTRL] = ctrl
    app[CLIENTS] = weakref.WeakSet()
    ctrl.listeners.append(lambda m: _send_all(app, m))

    app.router.add_get('/api/schema', get_schema)
    app.router.add_get('/api/config', get_config)
    app.router.add_patch('/api/config', patch_config)
    app.router.add_get('/api/presets', list_presets)
    app.router.add_post('/api/presets/{name}', save_preset)
    app.router.add_post('/api/presets/{name}/load', load_preset)
    app.router.add_delete('/api/presets/{name}', delete_preset)
    app.router.add_get('/api/info', get_info)
    app.router.add_get('/video.mjpg', video)
    app.router.add_get('/snapshot.jpg', snapshot)
    app.router.add_get('/ws', websocket)

    async def on_startup(app):
        if start_controller:
            await ctrl.start()
        app[TELEMETRY_TASK] = asyncio.create_task(_telemetry_loop(app))

    async def on_shutdown(app):
        for ws in list(app[CLIENTS]):
            await ws.close()

    async def on_cleanup(app):
        app[TELEMETRY_TASK].cancel()
        if start_controller:
            await ctrl.shutdown()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)
    return app
