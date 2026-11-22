"""Start the SignTrans backend and print the URL for the frontend .env."""

from __future__ import annotations

import os
import socket
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import uvicorn


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=1.5) as response:
            return response.status == 200
    except Exception:
        return False


def _print_url(url: str) -> None:
    print(f"Backend URL: {url}", flush=True)
    print(f"Frontend .env: VITE_API_BASE_URL={url}/api", flush=True)


def main() -> None:
    configured_port = os.getenv("PORT")
    if configured_port:
        port = int(configured_port)
        public_url = os.getenv("RENDER_EXTERNAL_URL") or f"http://localhost:{port}"
        _print_url(public_url)
        uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
        return

    host = "127.0.0.1"
    port = 8000
    url = f"http://localhost:{port}"
    if not _port_available(host, port):
        if _healthy(url):
            print("SignTrans backend is already running.", flush=True)
            _print_url(url)
            return
        for candidate in range(8001, 8011):
            if _port_available(host, candidate):
                port = candidate
                url = f"http://localhost:{port}"
                print(f"Port 8000 is busy; using {port}.", flush=True)
                break
        else:
            raise SystemExit("No free port found between 8000 and 8010.")

    os.environ.setdefault("PORT", str(port))
    _print_url(url)
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
