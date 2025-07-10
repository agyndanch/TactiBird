/**
 * TactiBird Overlay - Event Handler
 */

class EventHandler {
    constructor(app) {
        this.app = app;
        this.boundHandlers = new Map();
        
        console.log('Event Handler initialized');
    }
    
    setupUIEvents() {
        // Control button events
        this.addEventHandler('minimize-btn', 'click', () => {
            this.app.minimizeOverlay();
        });
        
        this.addEventHandler('close-btn', 'click', () => {
            this.app.closeOverlay();
        });
        
        this.addEventHandler('settings-btn', 'click', () => {
            this.app.openSettings();
        });
        
        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        console.log('UI events setup complete');
    }
    
    setupSettingsEvents() {
        // Settings modal events
        this.addEventHandler('close-settings', 'click', () => {
            this.app.uiManager.hideSettingsModal();
        });
        
        this.addEventHandler('save-settings', 'click', () => {
            this.saveSettings();
        });
        
        this.addEventHandler('cancel-settings', 'click', () => {
            this.app.uiManager.hideSettingsModal();
        });
        
        // Slider value updates
        this.setupSliderEvents();
        
        // Real-time setting previews
        this.setupSettingPreviews();
        
        console.log('Settings events setup complete');
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            // Ctrl+M - Minimize
            if (event.ctrlKey && event.key === 'm') {
                event.preventDefault();
                this.app.minimizeOverlay();
            }
            
            // Ctrl+, - Settings
            if (event.ctrlKey && event.key === ',') {
                event.preventDefault();
                this.app.openSettings();
            }
            
            // Escape - Close modal/minimize
            if (event.key === 'Escape') {
                const modal = document.getElementById('settings-modal');
                if (modal && !modal.classList.contains('hidden')) {
                    this.app.uiManager.hideSettingsModal();
                } else {
                    this.app.minimizeOverlay();
                }
            }
            
            // F5 - Refresh suggestions
            if (event.key === 'F5') {
                event.preventDefault();
                if (this.app.websocket && this.app.websocket.isConnected()) {
                    this.app.websocket.requestSuggestions();
                }
            }
        });
    }
    
    setupSliderEvents() {
        // Opacity slider
        const opacitySlider = document.getElementById('opacity-slider');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', (event) => {
                const opacity = parseFloat(event.target.value);
                this.previewOpacity(opacity);
            });
        }
        
        // Max suggestions slider
        const maxSuggestionsSlider = document.getElementById('max-suggestions');
        const maxSuggestionsValue = document.getElementById('max-suggestions-value');
        if (maxSuggestionsSlider && maxSuggestionsValue) {
            maxSuggestionsSlider.addEventListener('input', (event) => {
                maxSuggestionsValue.textContent = event.target.value;
            });
        }
        
        // Suggestion priority slider
        const prioritySlider = document.getElementById('suggestion-priority');
        const priorityValue = document.getElementById('suggestion-priority-value');
        if (prioritySlider && priorityValue) {
            prioritySlider.addEventListener('input', (event) => {
                priorityValue.textContent = event.target.value;
            });
        }
    }
    
    setupSettingPreviews() {
        // Theme preview
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect) {
            themeSelect.addEventListener('change', (event) => {
                this.previewTheme(event.target.value);
            });
        }
        
        // Always on top preview (if Electron API available)
        const alwaysOnTopCheckbox = document.getElementById('always-on-top');
        if (alwaysOnTopCheckbox) {
            alwaysOnTopCheckbox.addEventListener('change', (event) => {
                this.previewAlwaysOnTop(event.target.checked);
            });
        }
    }
    
    previewOpacity(opacity) {
        const container = document.querySelector('.overlay-container');
        if (container) {
            container.style.opacity = opacity;
        }
    }
    
    previewTheme(theme) {
        document.body.setAttribute('data-theme', theme);
    }
    
    previewAlwaysOnTop(alwaysOnTop) {
        if (window.electronAPI) {
            window.electronAPI.setAlwaysOnTop(alwaysOnTop);
        }
    }
    
    saveSettings() {
        try {
            const newSettings = this.app.uiManager.getSettingsFromForm();
            
            // Validate settings
            if (this.validateSettings(newSettings)) {
                // Apply settings
                this.app.updateSettings(newSettings);
                
                // Send to server if connected
                if (this.app.websocket && this.app.websocket.isConnected()) {
                    this.app.websocket.updateSettings(newSettings);
                }
                
                // Close modal
                this.app.uiManager.hideSettingsModal();
                
                // Show success notification
                this.app.showNotification('Settings saved successfully', 'success', 3000);
            }
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.app.showNotification('Failed to save settings', 'error', 5000);
        }
    }
    
    validateSettings(settings) {
        // Validate opacity
        if (settings.overlay && settings.overlay.opacity !== undefined) {
            const opacity = settings.overlay.opacity;
            if (opacity < 0.1 || opacity > 1.0) {
                this.app.showNotification('Opacity must be between 0.1 and 1.0', 'error');
                return false;
            }
        }
        
        // Validate max suggestions
        if (settings.suggestions && settings.suggestions.maxCount !== undefined) {
            const maxCount = settings.suggestions.maxCount;
            if (maxCount < 1 || maxCount > 20) {
                this.app.showNotification('Max suggestions must be between 1 and 20', 'error');
                return false;
            }
        }
        
        // Validate priority
        if (settings.suggestions && settings.suggestions.minPriority !== undefined) {
            const minPriority = settings.suggestions.minPriority;
            if (minPriority < 1 || minPriority > 10) {
                this.app.showNotification('Min priority must be between 1 and 10', 'error');
                return false;
            }
        }
        
        return true;
    }
    
    addEventHandler(elementId, event, handler) {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`Element not found: ${elementId}`);
            return;
        }
        
        // Store bound handler for cleanup
        const boundHandler = handler.bind(this);
        this.boundHandlers.set(`${elementId}-${event}`, {
            element,
            event,
            handler: boundHandler
        });
        
        element.addEventListener(event, boundHandler);
    }
    
    removeEventHandler(elementId, event) {
        const key = `${elementId}-${event}`;
        const handlerInfo = this.boundHandlers.get(key);
        
        if (handlerInfo) {
            handlerInfo.element.removeEventListener(handlerInfo.event, handlerInfo.handler);
            this.boundHandlers.delete(key);
        }
    }
    
    // Custom event system for internal communication
    createCustomEvent(name, detail = {}) {
        return new CustomEvent(name, {
            detail,
            bubbles: true,
            cancelable: true
        });
    }
    
    dispatchCustomEvent(name, detail = {}) {
        const event = this.createCustomEvent(name, detail);
        document.dispatchEvent(event);
        return event;
    }
    
    onCustomEvent(name, handler) {
        document.addEventListener(name, handler);
    }
    
    offCustomEvent(name, handler) {
        document.removeEventListener(name, handler);
    }
    
    // Utility methods for common event patterns
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
    
    // Handle window resize events
    setupResizeHandler() {
        const debouncedResize = this.debounce(() => {
            this.handleWindowResize();
        }, 250);
        
        window.addEventListener('resize', debouncedResize);
    }
    
    handleWindowResize() {
        // Adjust overlay positioning if needed
        const container = document.querySelector('.overlay-container');
        if (!container) return;
        
        const rect = container.getBoundingClientRect();
        const windowWidth = window.innerWidth;
        const windowHeight = window.innerHeight;
        
        // Keep overlay within screen bounds
        let newX = rect.left;
        let newY = rect.top;
        
        if (rect.right > windowWidth) {
            newX = windowWidth - rect.width - 10;
        }
        
        if (rect.bottom > windowHeight) {
            newY = windowHeight - rect.height - 10;
        }
        
        if (newX < 0) newX = 10;
        if (newY < 0) newY = 10;
        
        if (newX !== rect.left || newY !== rect.top) {
            container.style.left = `${newX}px`;
            container.style.top = `${newY}px`;
            container.style.right = 'auto';
            
            // Update settings
            this.app.updateSettings({
                overlay: {
                    ...this.app.settings.overlay,
                    position: { x: newX, y: newY }
                }
            });
        }
    }
    
    // Handle focus/blur for overlay behavior
    setupFocusHandlers() {
        window.addEventListener('focus', () => {
            this.dispatchCustomEvent('overlay-focus');
        });
        
        window.addEventListener('blur', () => {
            this.dispatchCustomEvent('overlay-blur');
        });
        
        // Handle visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.dispatchCustomEvent('overlay-hidden');
            } else {
                this.dispatchCustomEvent('overlay-visible');
            }
        });
    }
    
    // Context menu handling
    setupContextMenu() {
        document.addEventListener('contextmenu', (event) => {
            // Allow context menu on input elements
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return;
            }
            
            // Prevent default context menu on other elements
            event.preventDefault();
            
            // Could show custom context menu here
            this.showCustomContextMenu(event.clientX, event.clientY);
        });
    }
    
    showCustomContextMenu(x, y) {
        // Implementation for custom context menu
        // For now, just log the coordinates
        console.log(`Context menu requested at (${x}, ${y})`);
    }
    
    // Cleanup method
    cleanup() {
        // Remove all event handlers
        for (const [key, handlerInfo] of this.boundHandlers) {
            handlerInfo.element.removeEventListener(handlerInfo.event, handlerInfo.handler);
        }
        
        this.boundHandlers.clear();
        
        console.log('Event handlers cleaned up');
    }
}