/**
 * TactiBird Overlay - Electron Preload Script
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // Window controls
    minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
    closeWindow: () => ipcRenderer.invoke('close-window'),
    setAlwaysOnTop: (alwaysOnTop) => ipcRenderer.invoke('set-always-on-top', alwaysOnTop),
    
    // Window positioning
    setWindowPosition: (x, y) => ipcRenderer.invoke('set-window-position', x, y),
    getWindowPosition: () => ipcRenderer.invoke('get-window-position'),
    setWindowOpacity: (opacity) => ipcRenderer.invoke('set-window-opacity', opacity),
    
    // Display information
    getDisplayInfo: () => ipcRenderer.invoke('get-display-info'),
    
    // File operations
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    
    // External links
    openExternal: (url) => ipcRenderer.invoke('open-external', url),
    
    // App information
    getAppInfo: () => ipcRenderer.invoke('get-app-info'),
    
    // Event listeners
    onWindowClose: (callback) => {
        ipcRenderer.on('window-close', callback);
        return () => ipcRenderer.removeListener('window-close', callback);
    },
    
    onOpenSettings: (callback) => {
        ipcRenderer.on('open-settings', callback);
        return () => ipcRenderer.removeListener('open-settings', callback);
    },
    
    // Send events to main process
    sendToMain: (channel, ...args) => {
        const validChannels = [
            'window-moved',
            'settings-changed',
            'overlay-ready',
            'game-detected',
            'error-report'
        ];
        
        if (validChannels.includes(channel)) {
            ipcRenderer.send(channel, ...args);
        }
    },
    
    // Platform information
    platform: process.platform,
    versions: process.versions
});

// Expose utilities for the renderer
contextBridge.exposeInMainWorld('electronUtils', {
    // Path utilities
    joinPath: (...paths) => {
        const path = require('path');
        return path.join(...paths);
    },
    
    // OS utilities
    isWindows: () => process.platform === 'win32',
    isMacOS: () => process.platform === 'darwin',
    isLinux: () => process.platform === 'linux',
    
    // Performance utilities
    getMemoryUsage: () => process.memoryUsage(),
    getCPUUsage: () => process.cpuUsage(),
    
    // Environment
    isDevelopment: () => process.env.NODE_ENV === 'development',
    isProduction: () => process.env.NODE_ENV === 'production'
});

// Security: Remove dangerous globals
delete window.require;
delete window.exports;
delete window.module;

console.log('Preload script loaded');