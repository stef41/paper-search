"""
TCP proxy: accepts connections on Pilot port 80 and forwards them to the
ArXiv API service (api:8000).  Each accepted Pilot connection is bridged
bidirectionally with a plain TCP socket to the upstream.
"""

import os
import signal
import socket
import sys
import threading

from pilotprotocol import Driver

UPSTREAM_HOST = os.environ.get("UPSTREAM_HOST", "api")
UPSTREAM_PORT = int(os.environ.get("UPSTREAM_PORT", "8000"))
PILOT_PORT = int(os.environ.get("PILOT_PORT", "80"))
BUF_SIZE = 65536
FORWARD_TIMEOUT = 300  # max seconds a single direction can be idle
MAX_CONNECTIONS = 1024

_active_connections = threading.Semaphore(MAX_CONNECTIONS)
_shutdown = threading.Event()


def _forward(src, dst, label: str):
    """Copy bytes from src to dst until EOF or error."""
    try:
        while not _shutdown.is_set():
            data = src.read(BUF_SIZE) if hasattr(src, "read") else src.recv(BUF_SIZE)
            if not data:
                break
            if hasattr(dst, "write"):
                dst.write(data)
            else:
                dst.sendall(data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    except Exception as exc:
        print(f"[proxy] {label} error: {exc}", file=sys.stderr)
    finally:
        try:
            if hasattr(dst, "shutdown"):
                dst.shutdown(socket.SHUT_WR)
        except Exception:
            pass


def _handle(pilot_conn):
    """Bridge one Pilot connection to the upstream TCP service."""
    try:
        upstream = socket.create_connection((UPSTREAM_HOST, UPSTREAM_PORT), timeout=30)
    except Exception as exc:
        print(f"[proxy] upstream connect failed: {exc}", file=sys.stderr)
        pilot_conn.close()
        _active_connections.release()
        return

    t1 = threading.Thread(target=_forward, args=(pilot_conn, upstream, "pilot→upstream"), daemon=True)
    t2 = threading.Thread(target=_forward, args=(upstream, pilot_conn, "upstream→pilot"), daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=FORWARD_TIMEOUT)
    t2.join(timeout=FORWARD_TIMEOUT)
    for s in (upstream, pilot_conn):
        try:
            if hasattr(s, "shutdown"):
                s.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass
    _active_connections.release()


def main():
    def _sig_handler(signum, frame):
        print(f"[proxy] received signal {signum}, shutting down …", file=sys.stderr)
        _shutdown.set()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    with Driver() as d:
        info = d.info()
        print(f"[proxy] agent address: {info.get('address')}")
        print(f"[proxy] listening on pilot port {PILOT_PORT}, forwarding to {UPSTREAM_HOST}:{UPSTREAM_PORT}")

        with d.listen(PILOT_PORT) as listener:
            while not _shutdown.is_set():
                conn = listener.accept()
                if not _active_connections.acquire(blocking=False):
                    print("[proxy] connection limit reached, dropping", file=sys.stderr)
                    conn.close()
                    continue
                threading.Thread(target=_handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
