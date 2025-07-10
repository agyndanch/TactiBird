"""
TactiBird Overlay - WebSocket Server (Complete Fixed Version)
"""

import asyncio
import json
import logging
import websockets
from typing import Set, Dict, Any, List
from datetime import datetime
from src.utils.network import is_port_available, find_available_port
from src.ai.coaches.base_coach import CoachingSuggestion

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for overlay communication - Complete fixed version"""
    
    def __init__(self, port: int = 8765, host: str = "localhost"):
        """Initialize WebSocket server"""
        # Check if port is available first
        if not is_port_available(host, port):
            logger.warning(f"Port {port} is not available, trying to find alternative...")
            try:
                port = find_available_port(host, port)
                logger.info(f"Using alternative port: {port}")
            except RuntimeError as e:
                logger.error(f"Could not find available port: {e}")
                raise
            
        self.port = port
        self.host = host
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.running = False
        self.shutdown_event = asyncio.Event()
    
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self.clients)
    
    def is_running(self) -> bool:
        """Check if server is running - Fixed for websockets 11+"""
        try:
            # Check if server exists and is properly initialized
            if not self.running or self.server is None:
                return False
            
            # For websockets 11+, check if server is still serving
            # The is_closing() method was removed
            if hasattr(self.server, 'sockets'):
                return len(self.server.sockets) > 0
            
            # Fallback: just check if we have a server object and running flag
            return True
            
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
            return False
    
    def _is_websocket_closed(self, websocket) -> bool:
        """Check if websocket is closed - Compatible with websockets 11+"""
        try:
            # For websockets 11+, check state attribute
            if hasattr(websocket, 'state'):
                return websocket.state.name in ['CLOSED', 'CLOSING']
            # Fallback for older versions
            elif hasattr(websocket, 'closed'):
                return websocket.closed
            else:
                # If we can't determine, assume it's open
                return False
        except Exception:
            return True  # If there's an error, assume it's closed
    
    async def _send_to_client(self, websocket, data: Dict[str, Any]):
        """Send message to a specific client with error handling - Fixed for websockets 11+"""
        try:
            if self._is_websocket_closed(websocket):
                logger.warning("Attempted to send to closed websocket")
                return False
                
            message = json.dumps(data)
            await websocket.send(message)
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client {websocket.remote_address} disconnected during send")
            return False
        except Exception as e:
            logger.error(f"Failed to send message to client {websocket.remote_address}: {e}")
            return False
    
    async def broadcast_message(self, data: Dict[str, Any]):
        """Broadcast message to all connected clients - Fixed for websockets 11+"""
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected_clients = set()
        
        for client in self.clients.copy():  # Use copy to avoid modification during iteration
            try:
                if self._is_websocket_closed(client):
                    disconnected_clients.add(client)
                    continue
                    
                await client.send(message)
                logger.debug(f"Message sent to client {client.remote_address}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Client {client.remote_address} disconnected during broadcast")
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Failed to send message to client {client.remote_address}: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
        
        if disconnected_clients:
            logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
    
    async def broadcast_suggestions(self, suggestions: List[CoachingSuggestion]):
        """Broadcast coaching suggestions to all clients"""
        message = {
            "type": "suggestions",
            "timestamp": datetime.now().isoformat(),
            "suggestions": [suggestion.to_dict() for suggestion in suggestions],
            "count": len(suggestions)
        }
        
        await self.broadcast_message(message)
        logger.debug(f"Broadcasted {len(suggestions)} suggestions to {len(self.clients)} clients")
    
    async def broadcast_game_state(self, game_state):
        """Broadcast game state update to all clients"""
        message = {
            "type": "game_state",
            "timestamp": datetime.now().isoformat(),
            "data": game_state.to_dict()
        }
        
        await self.broadcast_message(message)
        logger.debug(f"Broadcasted game state to {len(self.clients)} clients")
    
    async def broadcast_system_message(self, message_type: str, content: str, priority: int = 5):
        """Broadcast system message to all clients"""
        message = {
            "type": "system_message",
            "timestamp": datetime.now().isoformat(),
            "message_type": message_type,
            "content": content,
            "priority": priority
        }
        
        await self.broadcast_message(message)
        logger.info(f"Broadcasted system message: {message_type} - {content}")
    
    async def _handle_client(self, websocket):
        """Handle new client connection - Fixed for websockets 11+"""
        client_id = "unknown"
        
        try:
            # Get client info safely
            try:
                client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            except Exception:
                client_id = "unknown_client"
                
            # Get path from websocket object (websockets 11+ change)
            path = getattr(websocket, 'path', '/')
            logger.info(f"New client connection: {client_id} (path: {path})")

            # Add client to set immediately
            self.clients.add(websocket)
            logger.info(f"Client added to set: {client_id} (Total clients: {len(self.clients)})")

            try:
                # Send welcome message
                welcome_msg = {
                    "type": "connection",
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "server_info": {
                        "version": "1.0.0",
                        "capabilities": ["suggestions", "game_state", "system_messages", "settings"],
                        "client_id": client_id,
                        "path": path
                    }
                }
                
                success = await self._send_to_client(websocket, welcome_msg)
                if not success:
                    logger.warning(f"Failed to send welcome message to {client_id}")
                    return
                
                logger.info(f"Welcome message sent to {client_id}")

                # Handle client messages in a loop
                async for message in websocket:
                    try:
                        await self._handle_message(websocket, message, client_id)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from {client_id}: {e}")
                        await self._send_to_client(websocket, {
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
                    except Exception as e:
                        logger.error(f"Error processing message from {client_id}: {e}")
                        # Continue handling other messages rather than breaking

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client {client_id} disconnected normally")
            except websockets.exceptions.ConnectionClosedError as e:
                logger.info(f"Client {client_id} disconnected unexpectedly: {e}")
            except asyncio.CancelledError:
                logger.info(f"Client {client_id} handler cancelled")
                raise  # Re-raise to properly handle cancellation
            except Exception as e:
                logger.error(f"Unexpected error handling client {client_id}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Fatal error in client handler for {client_id}: {e}", exc_info=True)
        finally:
            # Always clean up the client
            try:
                self.clients.discard(websocket)
                logger.info(f"Client {client_id} removed from set (Remaining: {len(self.clients)})")
                
                # Close websocket if still open - Fixed for websockets 11+
                if hasattr(websocket, 'close') and not self._is_websocket_closed(websocket):
                    await websocket.close()
                    logger.debug(f"Closed websocket for {client_id}")
                    
            except Exception as e:
                logger.error(f"Error during cleanup for {client_id}: {e}")
    
    async def _handle_message(self, websocket, message: str, client_id: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            logger.debug(f"Received message from {client_id}: {message_type}")
            
            if message_type == "ping":
                # Respond to ping with pong
                await self._send_to_client(websocket, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
            elif message_type == "get_suggestions":
                # Client requesting current suggestions
                logger.debug(f"Client {client_id} requested suggestions")
                # TODO: Implement suggestion retrieval
                
            elif message_type == "settings_update":
                # Client updating settings
                settings = data.get("settings", {})
                await self._handle_settings_update(settings)
                
            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}")
                await self._send_to_client(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}", exc_info=True)
            await self._send_to_client(websocket, {
                "type": "error",
                "message": "Internal server error"
            })
    
    async def _handle_settings_update(self, settings: Dict[str, Any]):
        """Handle settings update from client"""
        logger.info(f"Settings update received: {settings}")
        # TODO: Implement settings update logic
    
    async def start(self):
        """Start the WebSocket server with enhanced error handling"""
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

            # Create server with enhanced configuration
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                # Enhanced server configuration
                ping_interval=30,      # Send ping every 30 seconds
                ping_timeout=10,       # Wait 10 seconds for pong
                close_timeout=10,      # Wait 10 seconds for close
                max_size=2**20,        # 1MB max message size
                max_queue=32,          # Max queued messages per client
            )
            
            self.running = True
            logger.info(f"WebSocket server started successfully on {self.host}:{self.port}")

            # Wait a moment to ensure server is fully ready
            await asyncio.sleep(0.2)
            
            # Test server is actually running
            if not self.is_running():
                raise RuntimeError("Server failed to start properly")

        except OSError as e:
            if e.errno == 10048:  # Address already in use (Windows)
                logger.error(f"Port {self.port} is already in use. Please check if another instance is running or change the port in config.json")
            elif e.errno == 98:   # Address already in use (Linux)
                logger.error(f"Port {self.port} is already in use. Please check if another instance is running or change the port in config.json")
            else:
                logger.error(f"Failed to bind to {self.host}:{self.port} - {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the WebSocket server gracefully - Fixed for websockets 11+"""
        logger.info("Stopping WebSocket server...")
        self.running = False
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Close all client connections gracefully
            if self.clients:
                logger.info(f"Closing {len(self.clients)} client connections...")
                
                # Send shutdown notification
                shutdown_msg = {
                    "type": "system_message",
                    "message_type": "info",
                    "content": "Server shutting down",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Try to notify clients of shutdown
                await asyncio.gather(
                    *[self._send_to_client(client, shutdown_msg) for client in self.clients.copy()],
                    return_exceptions=True
                )
                
                # Give clients time to process shutdown message
                await asyncio.sleep(0.5)
                
                # Close all connections - Fixed for websockets 11+
                await asyncio.gather(
                    *[client.close() for client in self.clients.copy() if not self._is_websocket_closed(client)],
                    return_exceptions=True
                )
                
                self.clients.clear()
            
            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                logger.info("WebSocket server closed successfully")
        
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}", exc_info=True)
        
        logger.info("WebSocket server stopped")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get server health status"""
        return {
            "running": self.is_running(),
            "client_count": self.get_client_count(),
            "port": self.port,
            "host": self.host,
            "timestamp": datetime.now().isoformat()
        }