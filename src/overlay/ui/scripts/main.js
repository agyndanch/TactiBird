/**
 * TactiBird Overlay - Main Application Script
 */

class TFTOverlay {
    constructor() {
        this.websocket = null;
        this.uiManager = null;
        this.eventHandler = null;
        this.settings = this.loadSettings();
        this.gameState = null;
        this.suggestions = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        this.hasShownConnectedNotification = false;
        
        this.init();
    }
    
    async init() {
        console.log('Initializing TactiBird Overlay...');
        
        // Initialize components
        this.uiManager = new UIManager();
        this.eventHandler = new EventHandler(this);
        
        // Apply saved settings
        this.applySettings();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Connect to backend
        await this.connectWebSocket();
        
        // Hide loading indicator
        this.hideLoadingIndicator();
        
        console.log('TactiBird Overlay initialized successfully');
    }
    
    setupEventListeners() {
        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
        
        // UI events
        this.eventHandler.setupUIEvents();
        
        // Settings events
        this.eventHandler.setupSettingsEvents();
        
        // Drag and drop for repositioning
        this.setupDragAndDrop();
    }
    
    async connectWebSocket() {
        const wsUrl = `ws://localhost:${this.settings.overlay.port}`;
        
        try {
            console.log(`Attempting to connect to ${wsUrl}`);

            this.websocket = new WebSocketClient(wsUrl, {
                onOpen: () => this.onWebSocketOpen(),
                onMessage: (data) => this.onWebSocketMessage(data),
                onClose: () => this.onWebSocketClose(),
                onError: (error) => this.onWebSocketError(error)
            });

            // Add connection timeout
            const connectionPromise = this.websocket.connect();
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Connection timeout')), 10000);
            });

            await Promise.race([connectionPromise, timeoutPromise]);

        } catch (error) {
            console.error('Failed to connect to WebSocket:', error);
            this.showConnectionError();

            // If this is the initial connection attempt, try again after a delay
            if (this.reconnectAttempts === 0) {
                console.log('Retrying connection in 2 seconds...');
                setTimeout(() => this.attemptReconnect(), 2000);
            }
        }
    }
    
    onWebSocketOpen() {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.uiManager.updateConnectionStatus('connected');
        
        // Only show notification once, not on every reconnect
        if (!this.hasShownConnectedNotification) {
            this.showNotification('Connected to TactiBird', 'success');
            this.hasShownConnectedNotification = true;
        }
    }
    
    onWebSocketMessage(data) {
        try {
            switch (data.type) {
                case 'suggestions':
                    this.handleSuggestions(data.suggestions);
                    break;
                    
                case 'game_state':
                    this.handleGameState(data.data);
                    break;
                    
                case 'system_message':
                    this.handleSystemMessage(data);
                    break;
                    
                case 'pong':
                    console.log('Received pong from server');
                    break;
                    
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    }
    
    onWebSocketClose() {
        console.log('WebSocket connection closed');
        this.uiManager.updateConnectionStatus('disconnected');
        
        // Only attempt reconnection if we haven't exceeded max attempts
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.attemptReconnect();
        } else {
            console.error('Max reconnection attempts reached');
            this.showNotification(
                'Could not connect to TactiBird backend. Please restart the application.',
                'error',
                10000
            );
        }
    }
    
    onWebSocketError(error) {
        console.error('WebSocket error:', error);
        this.uiManager.updateConnectionStatus('disconnected');
    }
    
    async attemptReconnect() {
        this.reconnectAttempts++;
        this.uiManager.updateConnectionStatus('connecting');
        
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(async () => {
            try {
                await this.connectWebSocket();
            } catch (error) {
                console.error('Reconnection failed:', error);
            }
        }, this.reconnectDelay * this.reconnectAttempts);
    }
    
    handleSuggestions(suggestions) {
        console.log('Received suggestions:', suggestions);
        this.suggestions = suggestions;
        this.uiManager.updateSuggestions(suggestions);
        
        // Show high priority suggestions as notifications
        const highPrioritySuggestions = suggestions.filter(s => s.priority >= 8);
        highPrioritySuggestions.forEach(suggestion => {
            this.showNotification(suggestion.message, 'info', 5000);
        });
    }
    
    handleGameState(gameState) {
        console.log('Received game state:', gameState);
        this.gameState = gameState;
        this.uiManager.updateGameState(gameState);
        this.uiManager.updateEconomyPanel(gameState);
        this.uiManager.updateCompositionPanel(gameState);
    }
    
    handleSystemMessage(message) {
        console.log('System message:', message);
        
        const typeMap = {
            'info': 'info',
            'warning': 'warning',
            'error': 'error',
            'success': 'success'
        };
        
        const notificationType = typeMap[message.message_type] || 'info';
        this.showNotification(message.content, notificationType);
    }
    
    showNotification(message, type = 'info', duration = 3000) {
        this.uiManager.showNotification(message, type, duration);
    }
    
    showConnectionError() {
        this.uiManager.updateConnectionStatus('disconnected');
        
        // Show more helpful error message
        const errorMessage = this.reconnectAttempts === 0 
            ? 'Failed to connect to TactiBird backend. Please ensure the application is running.'
            : `Reconnection failed (${this.reconnectAttempts}/${this.maxReconnectAttempts}). Retrying...`;
        
        this.showNotification(errorMessage, 'error', 5000);
    }
    
    hideLoadingIndicator() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
    
    setupDragAndDrop() {
        const header = document.querySelector('.header');
        const container = document.querySelector('.overlay-container');
        let isDragging = false;
        let dragOffset = { x: 0, y: 0 };
        
        header.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragOffset.x = e.clientX - container.offsetLeft;
            dragOffset.y = e.clientY - container.offsetTop;
            
            document.addEventListener('mousemove', handleDrag);
            document.addEventListener('mouseup', handleDragEnd);
        });
        
        function handleDrag(e) {
            if (!isDragging) return;
            
            const newX = e.clientX - dragOffset.x;
            const newY = e.clientY - dragOffset.y;
            
            // Keep overlay within screen bounds
            const maxX = window.innerWidth - container.offsetWidth;
            const maxY = window.innerHeight - container.offsetHeight;
            
            const clampedX = Math.max(0, Math.min(newX, maxX));
            const clampedY = Math.max(0, Math.min(newY, maxY));
            
            container.style.left = `${clampedX}px`;
            container.style.top = `${clampedY}px`;
            container.style.right = 'auto';
        }
        
        function handleDragEnd() {
            isDragging = false;
            document.removeEventListener('mousemove', handleDrag);
            document.removeEventListener('mouseup', handleDragEnd);
            
            // Save new position
            const position = {
                x: container.offsetLeft,
                y: container.offsetTop
            };
            
            app.updateSettings({ overlay: { ...app.settings.overlay, position } });
        }
    }
    
    loadSettings() {
        localStorage.removeItem('tft-coach-settings');

        const defaultSettings = {
            overlay: {
                opacity: 0.9,
                alwaysOnTop: true,
                theme: 'dark',
                position: { x: 20, y: 20 },
                port: 8765
            },
            coaches: {
                economy: true,
                composition: true,
                positioning: true,
                items: true
            },
            suggestions: {
                maxCount: 5,
                minPriority: 5
            }
        };
        
        try {
            const saved = localStorage.getItem('tft-coach-settings');
            return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
        } catch (error) {
            console.warn('Failed to load settings:', error);
            return defaultSettings;
        }
    }
    
    saveSettings() {
        try {
            localStorage.setItem('tft-coach-settings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
        }
    }
    
    updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        this.saveSettings();
        this.applySettings();
    }
    
    applySettings() {
        // Apply opacity
        const container = document.querySelector('.overlay-container');
        if (container) {
            container.style.opacity = this.settings.overlay.opacity;
        }
        
        // Apply theme
        document.body.setAttribute('data-theme', this.settings.overlay.theme);
        
        // Apply position
        if (this.settings.overlay.position) {
            container.style.left = `${this.settings.overlay.position.x}px`;
            container.style.top = `${this.settings.overlay.position.y}px`;
            container.style.right = 'auto';
        }
        
        // Apply always on top (this would need Electron API)
        if (window.electronAPI) {
            window.electronAPI.setAlwaysOnTop(this.settings.overlay.alwaysOnTop);
        }
    }
    
    minimizeOverlay() {
        const container = document.querySelector('.overlay-container');
        container.classList.toggle('minimized');
    }
    
    closeOverlay() {
        if (window.electronAPI) {
            window.electronAPI.closeWindow();
        } else {
            window.close();
        }
    }
    
    openSettings() {
        this.uiManager.showSettingsModal(this.settings);
    }
    
    sendPing() {
        if (this.websocket && this.websocket.isConnected()) {
            this.websocket.send({
                type: 'ping',
                timestamp: new Date().toISOString()
            });
        }
    }
    
    cleanup() {
        console.log('Cleaning up TFT Overlay...');
        
        if (this.websocket) {
            this.websocket.disconnect();
        }
        
        // Save current settings
        this.saveSettings();
    }
}

// Global app instance
let app;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    app = new TFTOverlay();
    
    // Setup periodic ping
    setInterval(() => {
        if (app.websocket && app.websocket.isConnected()) {
            app.sendPing();
        }
    }, 30000); // Ping every 30 seconds
});

// Handle Electron-specific events
if (window.electronAPI) {
    window.electronAPI.onWindowClose(() => {
        app.cleanup();
    });
}

// Export for debugging
window.TFTOverlay = app;