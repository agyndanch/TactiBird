/**
 * TactiBird Overlay - WebSocket Client
 */

class WebSocketClient {
    constructor(url, options = {}) {
        this.url = url;
        this.options = {
            onOpen: options.onOpen || (() => {}),
            onMessage: options.onMessage || (() => {}),
            onClose: options.onClose || (() => {}),
            onError: options.onError || (() => {}),
            ...options
        };
        
        this.websocket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        this.messageQueue = [];
    }
    
    async connect() {
        try {
            console.log(`Connecting to WebSocket: ${this.url}`);
            
            this.websocket = new WebSocket(this.url);
            
            this.websocket.onopen = (event) => {
                console.log('WebSocket connected');
                this.connected = true;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                this.flushMessageQueue();
                this.options.onOpen(event);
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket disconnected');
                this.connected = false;
                this.stopHeartbeat();
                this.options.onClose(event);
                
                // Attempt reconnection if not a clean close
                if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.scheduleReconnect();
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.options.onError(error);
            };
            
            // Wait for connection to be established
            await this.waitForConnection();
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            throw error;
        }
    }
    
    waitForConnection(timeout = 5000) {
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            
            const checkConnection = () => {
                if (this.connected) {
                    resolve();
                } else if (Date.now() - startTime > timeout) {
                    reject(new Error('Connection timeout'));
                } else {
                    setTimeout(checkConnection, 100);
                }
            };
            
            checkConnection();
        });
    }
    
    handleMessage(data) {
        console.log('WebSocket message received:', data.type);
        
        switch (data.type) {
            case 'connection':
                this.handleConnectionMessage(data);
                break;

            case 'update':  // ADD THIS CASE
                this.options.onMessage(data);
                break;
                
            case 'suggestions':
                this.options.onMessage(data);
                break;
                
            case 'game_state':
                this.options.onMessage(data);
                break;
                
            case 'system_message':
                this.options.onMessage(data);
                break;
                
            case 'pong':
                this.handlePong(data);
                break;
                
            default:
                this.options.onMessage(data);
        }
    }
    
    handleConnectionMessage(data) {
        console.log('Connection established:', data);
        
        if (data.server_info) {
            console.log('Server capabilities:', data.server_info.capabilities);
        }
    }
    
    handlePong(data) {
        console.log('Received pong from server');
        
        // Calculate latency if timestamp is available
        if (data.timestamp) {
            const latency = Date.now() - new Date(data.timestamp).getTime();
            console.log(`Server latency: ${latency}ms`);
        }
    }
    
    send(data) {
        if (!this.connected || !this.websocket) {
            console.warn('WebSocket not connected, queuing message');
            this.messageQueue.push(data);
            return false;
        }
        
        try {
            const message = JSON.stringify(data);
            this.websocket.send(message);
            return true;
        } catch (error) {
            console.error('Failed to send WebSocket message:', error);
            return false;
        }
    }
    
    sendPing() {
        return this.send({
            type: 'ping',
            timestamp: new Date().toISOString()
        });
    }
    
    requestSuggestions() {
        return this.send({
            type: 'get_suggestions',
            timestamp: new Date().toISOString()
        });
    }
    
    updateSettings(settings) {
        return this.send({
            type: 'settings_update',
            settings: settings,
            timestamp: new Date().toISOString()
        });
    }
    
    flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.send(message);
        }
    }
    
    startHeartbeat() {
        this.stopHeartbeat(); // Clear any existing interval
        
        this.heartbeatInterval = setInterval(() => {
            if (this.connected) {
                this.sendPing();
            }
        }, 30000); // Send ping every 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
        
        console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
        
        setTimeout(() => {
            console.log(`Reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
            this.connect().catch(error => {
                console.error('Reconnection failed:', error);
            });
        }, delay);
    }
    
    disconnect() {
        console.log('Disconnecting WebSocket');
        
        this.connected = false;
        this.stopHeartbeat();
        
        if (this.websocket) {
            this.websocket.close(1000, 'Client disconnect');
            this.websocket = null;
        }
    }
    
    isConnected() {
        return this.connected && this.websocket && this.websocket.readyState === WebSocket.OPEN;
    }
    
    getReadyState() {
        if (!this.websocket) return 'DISCONNECTED';
        
        switch (this.websocket.readyState) {
            case WebSocket.CONNECTING:
                return 'CONNECTING';
            case WebSocket.OPEN:
                return 'OPEN';
            case WebSocket.CLOSING:
                return 'CLOSING';
            case WebSocket.CLOSED:
                return 'CLOSED';
            default:
                return 'UNKNOWN';
        }
    }
    
    getConnectionStats() {
        return {
            connected: this.connected,
            readyState: this.getReadyState(),
            reconnectAttempts: this.reconnectAttempts,
            queuedMessages: this.messageQueue.length,
            url: this.url
        };
    }
}