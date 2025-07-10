/**
 * TactiBird Overlay - UI Manager
 */

class UIManager {
    constructor() {
        this.elements = {};
        this.notifications = [];
        this.maxNotifications = 5;
        this.notificationDuration = 5000;
        
        this.initializeElements();
    }
    
    initializeElements() {
        // Cache frequently used elements
        this.elements = {
            // Status elements
            connectionStatus: document.getElementById('connection-status'),
            
            // Game state elements
            stageRound: document.getElementById('stage-round'),
            goldValue: document.getElementById('gold-value'),
            healthValue: document.getElementById('health-value'),
            levelValue: document.getElementById('level-value'),
            
            // Suggestions elements
            suggestionsList: document.getElementById('suggestions-list'),
            suggestionCount: document.getElementById('suggestion-count'),
            
            // Economy elements
            interestValue: document.getElementById('interest-value'),
            nextBreakpoint: document.getElementById('next-breakpoint'),
            economyStrength: document.getElementById('economy-strength'),
            
            // Composition elements
            activeTraits: document.getElementById('active-traits'),
            recommendedComp: document.getElementById('recommended-comp'),
            
            // Notifications
            notifications: document.getElementById('notifications'),
            
            // Modal elements
            settingsModal: document.getElementById('settings-modal')
        };
        
        console.log('UI Manager initialized');
    }
    
    updateConnectionStatus(status) {
        const statusElement = this.elements.connectionStatus;
        if (!statusElement) return;
        
        // Remove all status classes
        statusElement.classList.remove('connected', 'disconnected', 'connecting');
        
        // Add new status class
        statusElement.classList.add(status);
        
        console.log(`Connection status updated: ${status}`);
    }
    
    updateGameState(gameState) {
        try {
            // Update stage and round
            if (this.elements.stageRound && gameState.stage && gameState.round) {
                this.elements.stageRound.textContent = `${gameState.stage}-${gameState.round}`;
            }
            
            // Update player stats
            if (gameState.player) {
                this.updatePlayerStats(gameState.player);
            }
            
            console.log('Game state updated');
        } catch (error) {
            console.error('Failed to update game state:', error);
        }
    }
    
    updatePlayerStats(player) {
        // Update gold
        if (this.elements.goldValue && player.gold !== undefined) {
            this.elements.goldValue.textContent = player.gold;
            this.elements.goldValue.classList.add('gold');
        }
        
        // Update health with color coding
        if (this.elements.healthValue && player.health !== undefined) {
            this.elements.healthValue.textContent = player.health;
            
            // Color code based on health
            this.elements.healthValue.classList.remove('health-high', 'health-medium', 'health-low', 'health-critical');
            
            if (player.health > 60) {
                this.elements.healthValue.classList.add('health-high');
            } else if (player.health > 30) {
                this.elements.healthValue.classList.add('health-medium');
            } else if (player.health > 10) {
                this.elements.healthValue.classList.add('health-low');
            } else {
                this.elements.healthValue.classList.add('health-critical');
            }
        }
        
        // Update level
        if (this.elements.levelValue && player.level !== undefined) {
            this.elements.levelValue.textContent = player.level;
        }
    }
    
    updateSuggestions(suggestions) {
        const suggestionsList = this.elements.suggestionsList;
        const suggestionCount = this.elements.suggestionCount;
        
        if (!suggestionsList) return;
        
        // Update suggestion count
        if (suggestionCount) {
            suggestionCount.textContent = suggestions.length;
        }
        
        // Clear existing suggestions
        suggestionsList.innerHTML = '';
        
        if (suggestions.length === 0) {
            suggestionsList.innerHTML = `
                <div class="no-suggestions">
                    <p>No suggestions available</p>
                </div>
            `;
            return;
        }
        
        // Add new suggestions
        suggestions.forEach((suggestion, index) => {
            const suggestionElement = this.createSuggestionElement(suggestion, index);
            suggestionsList.appendChild(suggestionElement);
        });
        
        console.log(`Updated ${suggestions.length} suggestions`);
    }
    
    createSuggestionElement(suggestion, index) {
        const element = document.createElement('div');
        element.className = 'suggestion-item';
        
        // Add priority class
        const priority = suggestion.priority || 5;
        if (priority >= 8) {
            element.classList.add('priority-high');
        } else if (priority >= 6) {
            element.classList.add('priority-medium');
        } else {
            element.classList.add('priority-low');
        }
        
        // Add urgent class for very high priority
        if (priority >= 9) {
            element.classList.add('urgent');
        }
        
        element.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-type">${suggestion.type || 'general'}</span>
                <span class="suggestion-priority">${priority}</span>
            </div>
            <div class="suggestion-message">${suggestion.message}</div>
        `;
        
        // Add click handler for additional info
        element.addEventListener('click', () => {
            this.showSuggestionDetails(suggestion);
        });
        
        return element;
    }
    
    showSuggestionDetails(suggestion) {
        // Show detailed information about the suggestion
        const details = suggestion.context || {};
        let detailsText = `Type: ${suggestion.type}\nPriority: ${suggestion.priority}\n\n`;
        
        for (const [key, value] of Object.entries(details)) {
            detailsText += `${key}: ${value}\n`;
        }
        
        // For now, just log to console - could be enhanced with a modal
        console.log('Suggestion details:', detailsText);
    }
    
    updateEconomyPanel(gameState) {
        if (!gameState.player) return;
        
        const gold = gameState.player.gold || 0;
        const interest = Math.min(Math.floor(gold / 10), 5);
        
        // Update interest
        if (this.elements.interestValue) {
            this.elements.interestValue.textContent = interest;
        }
        
        // Update next breakpoint
        if (this.elements.nextBreakpoint) {
            const nextBreakpoint = this.calculateNextBreakpoint(gold);
            this.elements.nextBreakpoint.textContent = `${nextBreakpoint}g`;
        }
        
        // Update economy strength
        if (this.elements.economyStrength) {
            const strength = this.calculateEconomyStrength(gold);
            this.elements.economyStrength.textContent = strength.label;
            
            // Update class for color coding
            this.elements.economyStrength.className = `value economy-${strength.class}`;
        }
    }
    
    calculateNextBreakpoint(gold) {
        const breakpoints = [10, 20, 30, 40, 50];
        for (const breakpoint of breakpoints) {
            if (gold < breakpoint) {
                return breakpoint;
            }
        }
        return 50; // Max interest
    }
    
    calculateEconomyStrength(gold) {
        if (gold >= 50) {
            return { label: 'Excellent', class: 'excellent' };
        } else if (gold >= 30) {
            return { label: 'Strong', class: 'strong' };
        } else if (gold >= 20) {
            return { label: 'Decent', class: 'decent' };
        } else if (gold >= 10) {
            return { label: 'Weak', class: 'weak' };
        } else {
            return { label: 'Critical', class: 'critical' };
        }
    }
    
    updateCompositionPanel(gameState) {
        // Update active traits
        if (this.elements.activeTraits && gameState.board && gameState.board.active_traits) {
            this.updateActiveTraits(gameState.board.active_traits);
        }
        
        // Update recommended composition
        if (this.elements.recommendedComp) {
            this.updateRecommendedComp(gameState);
        }
    }
    
    updateActiveTraits(activeTraits) {
        const traitsContainer = this.elements.activeTraits;
        if (!traitsContainer) return;
        
        traitsContainer.innerHTML = '';
        
        if (!activeTraits || activeTraits.length === 0) {
            traitsContainer.innerHTML = '<p class="placeholder">No active traits</p>';
            return;
        }
        
        activeTraits.forEach(trait => {
            const traitElement = document.createElement('div');
            traitElement.className = 'trait-badge';
            
            if (trait.is_active || trait.active_level > 0) {
                traitElement.classList.add('active');
            }
            
            traitElement.innerHTML = `
                <span class="trait-name">${trait.name}</span>
                <span class="trait-count">${trait.current_count}/${trait.required_counts[0] || '?'}</span>
            `;
            
            traitsContainer.appendChild(traitElement);
        });
    }
    
    updateRecommendedComp(gameState) {
        const compContainer = this.elements.recommendedComp;
        if (!compContainer) return;
        
        // This would typically analyze the current composition and suggest improvements
        // For now, show a placeholder
        compContainer.innerHTML = '<p class="placeholder">Analyzing composition...</p>';
    }
    
    showNotification(message, type = 'info', duration = null) {
        const notificationElement = this.createNotificationElement(message, type);
        const container = this.elements.notifications;
        
        if (!container) {
            console.warn('Notifications container not found');
            return;
        }
        
        // Add to container
        container.appendChild(notificationElement);
        
        // Track notification
        const notification = {
            element: notificationElement,
            timestamp: Date.now(),
            duration: duration || this.notificationDuration
        };
        
        this.notifications.push(notification);
        
        // Remove old notifications if too many
        this.cleanupNotifications();
        
        // Auto-remove after duration
        setTimeout(() => {
            this.removeNotification(notification);
        }, notification.duration);
        
        // Add click to dismiss
        notificationElement.addEventListener('click', () => {
            this.removeNotification(notification);
        });
    }
    
    createNotificationElement(message, type) {
        const element = document.createElement('div');
        element.className = `notification ${type}`;
        element.textContent = message;
        return element;
    }
    
    removeNotification(notification) {
        if (notification.element && notification.element.parentNode) {
            notification.element.remove();
        }
        
        const index = this.notifications.indexOf(notification);
        if (index > -1) {
            this.notifications.splice(index, 1);
        }
    }
    
    cleanupNotifications() {
        while (this.notifications.length > this.maxNotifications) {
            const oldest = this.notifications.shift();
            this.removeNotification(oldest);
        }
    }
    
    showSettingsModal(currentSettings) {
        const modal = this.elements.settingsModal;
        if (!modal) return;
        
        // Populate settings with current values
        this.populateSettings(currentSettings);
        
        // Show modal
        modal.classList.remove('hidden');
    }
    
    hideSettingsModal() {
        const modal = this.elements.settingsModal;
        if (!modal) return;
        
        modal.classList.add('hidden');
    }
    
    populateSettings(settings) {
        // Populate opacity slider
        const opacitySlider = document.getElementById('opacity-slider');
        if (opacitySlider && settings.overlay && settings.overlay.opacity) {
            opacitySlider.value = settings.overlay.opacity;
        }
        
        // Populate always on top checkbox
        const alwaysOnTopCheckbox = document.getElementById('always-on-top');
        if (alwaysOnTopCheckbox && settings.overlay && settings.overlay.alwaysOnTop !== undefined) {
            alwaysOnTopCheckbox.checked = settings.overlay.alwaysOnTop;
        }
        
        // Populate theme selector
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect && settings.overlay && settings.overlay.theme) {
            themeSelect.value = settings.overlay.theme;
        }
        
        // Populate coach checkboxes
        if (settings.coaches) {
            const coachCheckboxes = {
                'economy-coach': settings.coaches.economy,
                'composition-coach': settings.coaches.composition,
                'positioning-coach': settings.coaches.positioning,
                'item-coach': settings.coaches.items
            };
            
            for (const [id, enabled] of Object.entries(coachCheckboxes)) {
                const checkbox = document.getElementById(id);
                if (checkbox && enabled !== undefined) {
                    checkbox.checked = enabled;
                }
            }
        }
        
        // Populate suggestion settings
        if (settings.suggestions) {
            const maxSuggestionsSlider = document.getElementById('max-suggestions');
            const maxSuggestionsValue = document.getElementById('max-suggestions-value');
            if (maxSuggestionsSlider && settings.suggestions.maxCount) {
                maxSuggestionsSlider.value = settings.suggestions.maxCount;
                if (maxSuggestionsValue) {
                    maxSuggestionsValue.textContent = settings.suggestions.maxCount;
                }
            }
            
            const prioritySlider = document.getElementById('suggestion-priority');
            const priorityValue = document.getElementById('suggestion-priority-value');
            if (prioritySlider && settings.suggestions.minPriority) {
                prioritySlider.value = settings.suggestions.minPriority;
                if (priorityValue) {
                    priorityValue.textContent = settings.suggestions.minPriority;
                }
            }
        }
    }
    
    getSettingsFromForm() {
        const settings = {
            overlay: {},
            coaches: {},
            suggestions: {}
        };
        
        // Get opacity
        const opacitySlider = document.getElementById('opacity-slider');
        if (opacitySlider) {
            settings.overlay.opacity = parseFloat(opacitySlider.value);
        }
        
        // Get always on top
        const alwaysOnTopCheckbox = document.getElementById('always-on-top');
        if (alwaysOnTopCheckbox) {
            settings.overlay.alwaysOnTop = alwaysOnTopCheckbox.checked;
        }
        
        // Get theme
        const themeSelect = document.getElementById('theme-select');
        if (themeSelect) {
            settings.overlay.theme = themeSelect.value;
        }
        
        // Get coach settings
        const coachSettings = {
            'economy-coach': 'economy',
            'composition-coach': 'composition',
            'positioning-coach': 'positioning',
            'item-coach': 'items'
        };
        
        for (const [id, key] of Object.entries(coachSettings)) {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                settings.coaches[key] = checkbox.checked;
            }
        }
        
        // Get suggestion settings
        const maxSuggestionsSlider = document.getElementById('max-suggestions');
        if (maxSuggestionsSlider) {
            settings.suggestions.maxCount = parseInt(maxSuggestionsSlider.value);
        }
        
        const prioritySlider = document.getElementById('suggestion-priority');
        if (prioritySlider) {
            settings.suggestions.minPriority = parseInt(prioritySlider.value);
        }
        
        return settings;
    }
    
    updateUI(element, value, className = null) {
        if (!this.elements[element]) return;
        
        this.elements[element].textContent = value;
        
        if (className) {
            this.elements[element].className = className;
        }
    }
    
    addClass(element, className) {
        if (!this.elements[element]) return;
        this.elements[element].classList.add(className);
    }
    
    removeClass(element, className) {
        if (!this.elements[element]) return;
        this.elements[element].classList.remove(className);
    }
    
    toggleClass(element, className) {
        if (!this.elements[element]) return;
        this.elements[element].classList.toggle(className);
    }
    
    setVisibility(element, visible) {
        if (!this.elements[element]) return;
        
        this.elements[element].style.display = visible ? '' : 'none';
    }
    
    animateElement(element, animation) {
        if (!this.elements[element]) return;
        
        this.elements[element].style.animation = animation;
        
        // Remove animation after it completes
        setTimeout(() => {
            if (this.elements[element]) {
                this.elements[element].style.animation = '';
            }
        }, 1000);
    }
    
    showLoadingState(element, loading = true) {
        if (!this.elements[element]) return;
        
        if (loading) {
            this.elements[element].classList.add('loading');
        } else {
            this.elements[element].classList.remove('loading');
        }
    }
    
    getElement(id) {
        return this.elements[id] || document.getElementById(id);
    }
    
    createElement(tag, className = '', content = '') {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (content) element.textContent = content;
        return element;
    }
    
    formatNumber(number, decimals = 0) {
        if (typeof number !== 'number') return number;
        return number.toFixed(decimals);
    }
    
    formatPercentage(value, total) {
        if (!total || total === 0) return '0%';
        return `${Math.round((value / total) * 100)}%`;
    }
    
    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
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
}