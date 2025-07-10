"""
TactiBird Overlay - Network Utilities
"""

import socket
import asyncio
import logging

logger = logging.getLogger(__name__)

def is_port_available(host: str = "localhost", port: int = 8765) -> bool:
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False

def find_available_port(host: str = "localhost", start_port: int = 8765, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

async def test_websocket_server(host: str = "localhost", port: int = 8765) -> bool:
    """Test if a WebSocket server is running on the given host:port"""
    try:
        import websockets
        async with websockets.connect(f"ws://{host}:{port}") as websocket:
            await websocket.ping()
            return True
    except Exception:
        return False