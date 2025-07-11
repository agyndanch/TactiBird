"""
Backend Integration for Playstyle Handling
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Any, Optional
from src.ai.coaches.economy_coach import EconomyCoach

logger = logging.getLogger(__name__)

class WebSocketServer:
    """WebSocket server for overlay communication with playstyle support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.port = config.get('overlay', {}).get('port', 8765)
        self.host = config.get('overlay', {}).get('host', 'localhost')
        
        # Initialize economy coach
        self.economy_coach = EconomyCoach(config.get('economy_coach', {}))
        
        # Connected clients
        self.clients = set()
        
        # Current game state
        self.current_game_state = None
        
        # Server instance
        self.server = None
        
        logger.info(f"WebSocket server initialized on {self.host}:{self.port}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            return self.server
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def handle_client(self, websocket):
        """Handle a new client connection - FIXED: Removed 'path' parameter"""
        try:
            self.clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            # Send connection confirmation
            await self.send_to_client(websocket, {
                'type': 'connection',
                'status': 'connected',
                'server_info': {
                    'capabilities': ['suggestions', 'game_state', 'playstyle_change']
                }
            })
            
            # Send initial game state if available
            if self.current_game_state:
                await self.send_update_to_client(websocket, self.current_game_state)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    })
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.send_to_client(websocket, {
                        'type': 'error',
                        'message': 'Message processing error'
                    })
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_message(self, websocket, data: Dict[str, Any]):
        """Handle incoming messages from clients"""
        try:
            message_type = data.get('type')
            logger.debug(f"Received message type: {message_type}")
            
            if message_type == 'playstyle_change':
                await self.handle_playstyle_change(websocket, data)
            elif message_type == 'request_suggestions':
                await self.handle_suggestion_request(websocket, data)
            elif message_type == 'ping':
                await self.handle_ping(websocket, data)
            elif message_type == 'settings_update':
                await self.handle_settings_update(websocket, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send_to_client(websocket, {
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                })
                
        except Exception as e:
            logger.error(f"Error handling message type {data.get('type')}: {e}")
    
    async def handle_ping(self, websocket, data: Dict[str, Any]):
        """Handle ping messages"""
        try:
            response = {
                'type': 'pong',
                'timestamp': data.get('timestamp')
            }
            await self.send_to_client(websocket, response)
        except Exception as e:
            logger.error(f"Error handling ping: {e}")
    
    async def handle_settings_update(self, websocket, data: Dict[str, Any]):
        """Handle settings update messages"""
        try:
            settings = data.get('settings', {})
            logger.info(f"Settings updated: {settings}")
            
            # You can update configuration here if needed
            # self.config.update(settings)
            
            response = {
                'type': 'settings_response',
                'status': 'updated'
            }
            await self.send_to_client(websocket, response)
        except Exception as e:
            logger.error(f"Error handling settings update: {e}")
    
    async def handle_playstyle_change(self, websocket, data: Dict[str, Any]):
        """Handle playstyle change from overlay"""
        try:
            playstyle = data.get('playstyle')
            logger.info(f"Playstyle changed to: {playstyle}")
            
            # Update economy coach with new playstyle
            if hasattr(self.economy_coach, 'set_playstyle'):
                self.economy_coach.set_playstyle(playstyle)
            
            # Generate new suggestions if we have current game state
            if self.current_game_state:
                suggestions = await self.economy_coach.get_suggestions(self.current_game_state)
                
                # Send updated suggestions to client
                update_data = {
                    'type': 'update',
                    'stats': self._format_stats(getattr(self.current_game_state, 'stats', None)),
                    'economy': self._format_economy(self.current_game_state),
                    'suggestions': suggestions
                }
                
                await self.send_to_client(websocket, update_data)
            
        except Exception as e:
            logger.error(f"Error handling playstyle change: {e}")
    
    async def handle_suggestion_request(self, websocket, data: Dict[str, Any]):
        """Handle explicit suggestion requests"""
        try:
            if self.current_game_state:
                suggestions = await self.economy_coach.get_suggestions(self.current_game_state)
                
                response = {
                    'type': 'suggestions_response',
                    'suggestions': suggestions
                }
                await self.send_to_client(websocket, response)
            else:
                # No game state available
                response = {
                    'type': 'suggestions_response',
                    'suggestions': [],
                    'message': 'No game state available'
                }
                await self.send_to_client(websocket, response)
                
        except Exception as e:
            logger.error(f"Error handling suggestion request: {e}")
    
    async def update_game_state(self, game_state):
        """Update game state and broadcast to all clients"""
        try:
            self.current_game_state = game_state
            
            # Generate suggestions using economy coach
            suggestions = await self.economy_coach.get_suggestions(game_state)
            
            # Format update data
            update_data = {
                'type': 'update',
                'stats': self._format_stats(getattr(game_state, 'stats', None)),
                'economy': self._format_economy(game_state),
                'suggestions': suggestions,
                'timestamp': getattr(game_state, 'timestamp', None) if hasattr(game_state, 'timestamp') else getattr(game_state, 'timestamp', None)
            }
            
            # Broadcast to all connected clients
            await self.broadcast_to_all_clients(update_data)
            
        except Exception as e:
            logger.error(f"Error updating game state: {e}")
    
    async def send_to_client(self, websocket, data: Dict[str, Any]):
        """Send data to a specific client"""
        try:
            if websocket in self.clients:
                await websocket.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
            logger.debug("Client connection closed during send")
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self.clients.discard(websocket)
    
    async def send_update_to_client(self, websocket, game_state):
        """Send current game state to a specific client"""
        try:
            suggestions = await self.economy_coach.get_suggestions(game_state)
            
            update_data = {
                'type': 'update',
                'stats': self._format_stats(getattr(game_state, 'stats', None)),
                'economy': self._format_economy(game_state),
                'suggestions': suggestions,
                'timestamp': getattr(game_state, 'timestamp', None) if hasattr(game_state, 'timestamp') else getattr(game_state, 'timestamp', None)
            }
            
            await self.send_to_client(websocket, update_data)
            
        except Exception as e:
            logger.error(f"Error sending update to client: {e}")
    
    async def broadcast_to_all_clients(self, data: Dict[str, Any]):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        disconnected_clients = []
        
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(data))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)
    
    def _format_stats(self, stats) -> Dict[str, Any]:
        """Format stats for overlay"""
        if not stats:
            return {
                'gold': None,
                'health': None,
                'level': None,
                'stage': None,
                'round': None,
                'confidence': {}
            }
        
        # Handle both dict and object stats
        if hasattr(stats, '__dict__'):
            # Object with attributes
            return {
                'gold': getattr(stats, 'gold', None),
                'health': getattr(stats, 'health', None),
                'level': getattr(stats, 'level', None),
                'stage': getattr(stats, 'stage', None),
                'round': getattr(stats, 'round_num', None),
                'confidence': {
                    'gold': getattr(stats, 'gold_confidence', 0.0),
                    'health': getattr(stats, 'health_confidence', 0.0)
                }
            }
        else:
            # Dictionary
            return {
                'gold': stats.get('gold'),
                'health': stats.get('health'),
                'level': stats.get('level'),
                'stage': stats.get('stage'),
                'round': stats.get('round_num'),
                'confidence': {
                    'gold': stats.get('gold_confidence', 0.0),
                    'health': stats.get('health_confidence', 0.0)
                }
            }
    
    def _format_economy(self, game_state) -> Dict[str, Any]:
        """Format economy data for overlay"""
        if not game_state:
            return {}

        # Handle both object and dictionary game_state
        if hasattr(game_state, 'stats'):
            stats = game_state.stats
        else:
            stats = getattr(game_state, 'stats', None)

        if not stats:
            return {}
        
        # Extract gold value
        gold = None
        if hasattr(stats, 'gold'):
            gold = stats.gold
        elif isinstance(stats, dict):
            gold = stats.get('gold')
        
        stage = self._get_stage(stats)
        
        # Define thresholds based on game stage
        thresholds = {
            1: {'excellent': 25, 'strong': 20, 'decent': 15, 'weak': 10},
            2: {'excellent': 35, 'strong': 30, 'decent': 25, 'weak': 15},
            3: {'excellent': 45, 'strong': 40, 'decent': 30, 'weak': 20},
            4: {'excellent': 55, 'strong': 50, 'decent': 40, 'weak': 25},
            5: {'excellent': 65, 'strong': 55, 'decent': 45, 'weak': 30}
        }
        
        # Use stage 5 thresholds for stages 6+
        stage_thresholds = thresholds.get(min(stage, 5), thresholds[5])
        
        if gold is not None:
            if gold >= stage_thresholds['excellent']:
                economy_status = 'excellent'
            elif gold >= stage_thresholds['strong']:
                economy_status = 'strong'
            elif gold >= stage_thresholds['decent']:
                economy_status = 'decent'
            elif gold >= stage_thresholds['weak']:
                economy_status = 'weak'
            else:
                economy_status = 'critical'
        else:
            economy_status = 'unknown'
        
        return {
            'gold': gold,
            'stage': stage,
            'status': economy_status,
            'interest': self._calculate_interest(gold),
            'thresholds': stage_thresholds
        }
    
    def _get_stage(self, stats) -> int:
        """Extract stage from stats"""
        if hasattr(stats, 'stage'):
            return getattr(stats, 'stage', 1)
        elif isinstance(stats, dict):
            return stats.get('stage', 1)
        return 1
    
    def _calculate_interest(self, gold: Optional[int]) -> Optional[int]:
        """Calculate interest income"""
        if gold is None:
            return None
        
        # TFT interest: 1 gold per 10 gold, max 5
        return min(gold // 10, 5)
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        try:
            # Close all client connections
            if self.clients:
                logger.info(f"Closing {len(self.clients)} client connections...")
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
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
    
    def get_connection_count(self) -> int:
        """Get the number of connected clients"""
        return len(self.clients)
    
    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.server is not None and not self.server.is_serving()