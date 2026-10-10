#!/usr/bin/env bash
# Start/stop turretd on the Pi without systemd.
#   scripts/turretd.sh start [--sim]   start in the background (log: /tmp/turretd.log)
#   scripts/turretd.sh stop            SIGTERM → drivers disabled, ESCs stopped
#   scripts/turretd.sh restart | status | log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE=/tmp/turretd.pid
LOG=/tmp/turretd.log

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

start() {
  if running; then echo "already running (pid $(cat "$PIDFILE"))"; return 0; fi
  cd "$ROOT/src" || exit 1
  setsid nohup python3 -m turretd "$@" >"$LOG" 2>&1 </dev/null &
  echo $! >"$PIDFILE"
  sleep 2
  if running; then echo "started (pid $(cat "$PIDFILE")), log: $LOG"
  else echo "failed to start:"; tail -20 "$LOG"; return 1; fi
}

stop() {
  if ! running; then echo "not running"; rm -f "$PIDFILE"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  kill -TERM "$pid"
  for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then echo "did not stop in 10 s, killing"; kill -KILL "$pid"; fi
  rm -f "$PIDFILE"
  echo "stopped"
}

case "${1:-}" in
  start)   shift; start "$@" ;;
  stop)    stop ;;
  restart) shift; stop; start "$@" ;;
  status)  if running; then echo "running (pid $(cat "$PIDFILE"))"; else echo "not running"; fi ;;
  log)     tail -n 50 -f "$LOG" ;;
  *)       echo "usage: $0 start [--sim] | stop | restart | status | log"; exit 2 ;;
esac
