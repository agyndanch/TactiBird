"""
TactiBird Overlay - WebSocket Server (Fixed)
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
    """WebSocket server for overlay communication"""
    
    def __init__(self, port: int = 8765, host: str = "localhost"):
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
    
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def broadcast_message(self, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(data)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients
    
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
    
    async def _handle_settings_update(self, settings: Dict[str, Any]):
        """Handle settings update from client"""
        # TODO: Implement settings update logic
        # This could update overlay appearance, coach preferences, etc.
        logger.info(f"Settings update received: {settings}")
    
    def get_client_count(self) -> int:
        """Get number of connected clients"""
        return len(self.clients)
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running and self.server is not None
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                # Add server configuration for better stability
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.running = True
            logger.info(f"WebSocket server started successfully on {self.host}:{self.port}")

            # Wait a moment to ensure server is fully ready
            await asyncio.sleep(0.1)

        except OSError as e:
            if e.errno == 10048:  # Address already in use (Windows)
                logger.error(f"Port {self.port} is already in use. Please check if another instance is running or change the port in config.json")
            elif e.errno == 98:   # Address already in use (Linux)
                logger.error(f"Port {self.port} is already in use. Please check if another instance is running or change the port in config.json")
            else:
                logger.error(f"Failed to bind to {self.host}:{self.port} - {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
            self.clients.clear()
        
        # Close the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_client(self, websocket, path):
        """Handle new client connection"""
        try:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"Client connected: {client_id}")

            # Add client to set
            self.clients.add(websocket)

            try:
                # Send welcome message
                welcome_msg = {
                    "type": "connection",
                    "status": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "server_info": {
                        "version": "1.0.0",
                        "capabilities": ["suggestions", "game_state", "settings"]
                    }
                }
                await self._send_to_client(websocket, welcome_msg)

                # Handle client messages
                async for message in websocket:
                    try:
                        await self._handle_message(websocket, message)
                    except Exception as e:
                        logger.error(f"Error processing message from {client_id}: {e}")
                        # Continue handling other messages

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected cleanly: {client_id}")
            except websockets.exceptions.ConnectionClosedError:
                logger.info(f"Client disconnected unexpectedly: {client_id}")
            except Exception as e:
                logger.error(f"Error handling client {client_id}: {e}")
            finally:
                # Remove client from set
                self.clients.discard(websocket)
                logger.debug(f"Client {client_id} removed from client list")

        except Exception as e:
            logger.error(f"Fatal error in client handler: {e}")
            # Try to close the websocket if it's still open
            try:
                if not websocket.closed:
                    await websocket.close()
            except:
                pass
    
    async def _handle_message(self, websocket, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                await self._send_to_client(websocket, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "get_suggestions":
                # Client requesting current suggestions
                await self._send_to_client(websocket, {
                    "type": "suggestions_request_acknowledged",
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "settings_update":
                # Handle settings updates
                settings = data.get("settings", {})
                await self._handle_settings_update(settings)
                
                await self._send_to_client(websocket, {
                    "type": "settings_updated",
                    "timestamp": datetime.now().isoformat()
                })
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from client")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _send_to_client(self, websocket, data: Dict[str, Any]):
        """Send data to specific client"""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")