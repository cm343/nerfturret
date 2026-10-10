"""
Run the turret control service.

    cd src && python3 -m turretd              # on the Pi: real GPIO, ESCs, camera
    cd src && python3 -m turretd --sim        # anywhere: simulated hardware + camera

Then point the laptop UI at it (ui/turret_ui.py --pi <host>:8090).
"""

import argparse
import logging

from aiohttp import web

from .camera import CameraService
from .config import DEFAULT_CONFIG_DIR, ConfigStore
from .controller import AXES, AXIS_PINS, Controller
from .flywheels import Flywheels, kill_hwpwm, make_output
from .motion import MotionClient
from .server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog='turretd', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sim', action='store_true', help='simulate all hardware')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--config-dir', default=DEFAULT_CONFIG_DIR)
    ap.add_argument('--no-camera', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('turretd')

    cfg = ConfigStore(args.config_dir)
    for problem in cfg.load_errors:
        log.warning('config: %s', problem)

    # Fork the motion worker before anything starts threads (camera, lgpio).
    esc_backend = 'sim' if args.sim else cfg['flywheel.backend']
    motion = MotionClient([AXIS_PINS[a] for a in AXES], 'sim' if args.sim else 'gpio',
                          on_orphan=kill_hwpwm if esc_backend == 'hwpwm' else None)
    motion.start()

    fly = Flywheels(make_output(esc_backend),
                    min_us=cfg['flywheel.esc_min_us'], max_us=cfg['flywheel.esc_max_us'])

    camera = None
    ctrl_ref: list[Controller] = []
    if not args.no_camera:
        def view():
            c = ctrl_ref[0] if ctrl_ref else None
            if c is None:
                return 0.0, 0.0
            return c.position_deg('pan'), c.position_deg('tilt')
        camera = CameraService('sim' if args.sim else 'picam', view=view)

    ctrl = Controller(cfg, motion, fly, camera)
    ctrl_ref.append(ctrl)
    if camera:
        camera.start(cfg.snapshot())
    ctrl.listeners.append(lambda m: m.get('type') == 'event'
                          and log.log(logging.ERROR if m['level'] == 'error' else logging.INFO,
                                      m['msg']))

    app = create_app(ctrl)

    async def cleanup(_app):
        if camera:
            camera.stop()
        fly.close()
        motion.stop()
    app.on_cleanup.append(cleanup)

    log.info('turretd on http://%s:%d  (motion=%s, camera=%s)', args.host, args.port,
             motion.backend, camera.backend if camera else 'off')
    # Short shutdown: open video streams must not delay stopping the hardware.
    web.run_app(app, host=args.host, port=args.port, print=None,
                handle_signals=True, shutdown_timeout=2.0)


if __name__ == '__main__':
    main()
