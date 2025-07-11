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
        
        logger.info(f"WebSocket server initialized on {self.host}:{self.port}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        try:
            server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            return server
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def handle_client(self, websocket, path):
        """Handle a new client connection"""
        try:
            self.clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
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
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
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
            
            if message_type == 'playstyle_change':
                await self.handle_playstyle_change(websocket, data)
            elif message_type == 'request_suggestions':
                await self.handle_suggestion_request(websocket, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message type {data.get('type')}: {e}")
    
    async def handle_playstyle_change(self, websocket, data: Dict[str, Any]):
        """Handle playstyle change from overlay"""
        try:
            playstyle = data.get('playstyle')
            logger.info(f"Playstyle changed to: {playstyle}")
            
            # Update economy coach with new playstyle
            self.economy_coach.set_playstyle(playstyle)
            
            # Generate new suggestions if we have current game state
            if self.current_game_state:
                suggestions = await self.economy_coach.get_suggestions(self.current_game_state)
                
                # Send updated suggestions to client
                update_data = {
                    'type': 'update',
                    'stats': self._format_stats(self.current_game_state.stats),
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
                    'suggestions': []
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
                'stats': self._format_stats(game_state.stats),
                'economy': self._format_economy(game_state),
                'suggestions': suggestions
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
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
    
    async def send_update_to_client(self, websocket, game_state):
        """Send current game state to a specific client"""
        try:
            suggestions = await self.economy_coach.get_suggestions(game_state)
            
            update_data = {
                'type': 'update',
                'stats': self._format_stats(game_state.stats),
                'economy': self._format_economy(game_state),
                'suggestions': suggestions
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
        
        return {
            'gold': stats.gold,
            'health': stats.health,
            'level': stats.level,
            'stage': stats.stage,
            'round': stats.round_num,
            'confidence': {
                'gold': getattr(stats, 'gold_confidence', 0.0),
                'health': getattr(stats, 'health_confidence', 0.0)
            }
        }
    
    def _format_economy(self, game_state) -> Dict[str, Any]:
        """Format economy data for overlay"""
        if not game_state or not game_state.stats:
            return {
                'economy_strength': 'unknown',
                'interest': None,
                'streak': None
            }
        
        # Calculate economy strength based on gold and stage
        economy_strength = self._calculate_economy_strength(game_state.stats)
        
        return {
            'economy_strength': economy_strength,
            'interest': self._calculate_interest(game_state.stats.gold),
            'streak': getattr(game_state.stats, 'streak', None)
        }
    
    def _calculate_economy_strength(self, stats) -> str:
        """Calculate economy strength classification"""
        if not stats or stats.gold is None:
            return 'unknown'
        
        gold = stats.gold
        stage = stats.stage or 1
        
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
        
        if gold >= stage_thresholds['excellent']:
            return 'excellent'
        elif gold >= stage_thresholds['strong']:
            return 'strong'
        elif gold >= stage_thresholds['decent']:
            return 'decent'
        elif gold >= stage_thresholds['weak']:
            return 'weak'
        else:
            return 'critical'
    
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
                await asyncio.gather(
                    *[client.close() for client in self.clients],
                    return_exceptions=True
                )
            
            logger.info("WebSocket server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")


# Integration with main application
"""
Example of how to integrate this into your main application:

File: src/main.py or wherever your main application loop is
"""

class TactiBirdApplication:
    """Main application class with integrated overlay server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.overlay_server = OverlayWebSocketServer(config)
        # ... other components
    
    async def start(self):
        """Start the application"""
        try:
            # Start WebSocket server
            server = await self.overlay_server.start_server()
            
            # Start other components (OCR, game detection, etc.)
            await self.start_game_components()
            
            # Keep running
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            raise
    
    async def start_game_components(self):
        """Start game detection and processing components"""
        # This is where you'd start your existing game state detection
        # and processing logic
        pass
    
    async def on_game_state_update(self, game_state):
        """Called when game state is updated"""
        try:
            # Update overlay with new game state
            await self.overlay_server.update_game_state(game_state)
            
        except Exception as e:
            logger.error(f"Error updating game state: {e}")


# Example usage in your existing game loop
"""
When you detect game state changes in your existing code, call:

await app.overlay_server.update_game_state(new_game_state)

This will automatically:
1. Generate suggestions using the economy coach
2. Broadcast updates to all connected overlay clients
3. Include playstyle-specific suggestions if a playstyle is selected
"""