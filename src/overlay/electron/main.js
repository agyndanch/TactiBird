/**
 * TactiBird Overlay - Electron Main Process
 */

const { app, BrowserWindow, ipcMain, screen, Menu, Tray, dialog, shell } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';

class TactibirdElectronApp {
    constructor() {
        this.mainWindow = null;
        this.tray = null;
        this.isQuitting = false;
        
        // Initialize app
        this.initializeApp();
    }
    
    initializeApp() {
        // Handle app ready
        app.whenReady().then(() => {
            this.createMainWindow();
            this.createTray();
            this.setupIPC();
            this.setupMenu();
            
            // macOS specific behavior
            app.on('activate', () => {
                if (BrowserWindow.getAllWindows().length === 0) {
                    this.createMainWindow();
                }
            });
        });
        
        // Handle window close
        app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                app.quit();
            }
        });
        
        // Handle app quit
        app.on('before-quit', () => {
            this.isQuitting = true;
        });
        
        // Security: Prevent new window creation
        app.on('web-contents-created', (event, contents) => {
            contents.on('new-window', (event, navigationUrl) => {
                event.preventDefault();
                shell.openExternal(navigationUrl);
            });
        });
    }
    
    createMainWindow() {
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workAreaSize;
        
        // Create the browser window
        this.mainWindow = new BrowserWindow({
            width: 380,
            height: 600,
            x: width - 400,
            y: 50,
            frame: false,
            transparent: true,
            alwaysOnTop: true,
            resizable: false,
            skipTaskbar: true,
            webPreferences: {
                nodeIntegration: false,
                contextIsolation: true,
                enableRemoteModule: false,
                preload: path.join(__dirname, 'preload.js')
            },
            icon: path.join(__dirname, '../../assets/icons/app_icon.png')
        });
        
        // Load the overlay
        const startUrl = isDev 
            ? 'http://localhost:8080' 
            : `file://${path.join(__dirname, '../ui/templates/overlay.html')}`;
        
        this.mainWindow.loadURL(startUrl);
        
        // Open DevTools in development
        if (isDev) {
            this.mainWindow.webContents.openDevTools({ mode: 'detach' });
        }
        
        // Handle window close
        this.mainWindow.on('close', (event) => {
            if (!this.isQuitting) {
                event.preventDefault();
                this.mainWindow.hide();
            }
        });
        
        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
        });
        
        // Handle window ready
        this.mainWindow.webContents.once('ready-to-show', () => {
            if (isDev) {
                this.mainWindow.show();
            }
        });
        
        console.log('Main window created');
    }
    
    createTray() {
        const iconPath = path.join(__dirname, '../../assets/icons/tray_icon.png');
        this.tray = new Tray(iconPath);
        
        const contextMenu = Menu.buildFromTemplate([
            {
                label: 'Show/Hide TactiBird',
                click: () => this.toggleWindow()
            },
            { type: 'separator' },
            {
                label: 'Settings',
                click: () => this.openSettings()
            },
            {
                label: 'About',
                click: () => this.showAbout()
            },
            { type: 'separator' },
            {
                label: 'Quit',
                click: () => this.quitApp()
            }
        ]);
        
        this.tray.setToolTip('TactiBird - TFT AI Coach');
        this.tray.setContextMenu(contextMenu);
        
        // Handle tray click
        this.tray.on('click', () => {
            this.toggleWindow();
        });
        
        console.log('System tray created');
    }
    
    setupIPC() {
        // Handle window control requests
        ipcMain.handle('minimize-window', () => {
            if (this.mainWindow) {
                this.mainWindow.minimize();
            }
        });
        
        ipcMain.handle('close-window', () => {
            if (this.mainWindow) {
                this.mainWindow.hide();
            }
        });
        
        ipcMain.handle('set-always-on-top', (event, alwaysOnTop) => {
            if (this.mainWindow) {
                this.mainWindow.setAlwaysOnTop(alwaysOnTop);
            }
        });
        
        // Handle positioning
        ipcMain.handle('set-window-position', (event, x, y) => {
            if (this.mainWindow) {
                this.mainWindow.setPosition(x, y);
            }
        });
        
        ipcMain.handle('get-window-position', () => {
            if (this.mainWindow) {
                return this.mainWindow.getPosition();
            }
            return [0, 0];
        });
        
        // Handle opacity
        ipcMain.handle('set-window-opacity', (event, opacity) => {
            if (this.mainWindow) {
                this.mainWindow.setOpacity(opacity);
            }
        });
        
        // Handle display info
        ipcMain.handle('get-display-info', () => {
            const displays = screen.getAllDisplays();
            const primaryDisplay = screen.getPrimaryDisplay();
            
            return {
                displays: displays.map(display => ({
                    id: display.id,
                    bounds: display.bounds,
                    workArea: display.workArea,
                    scaleFactor: display.scaleFactor,
                    primary: display.id === primaryDisplay.id
                })),
                primary: {
                    id: primaryDisplay.id,
                    bounds: primaryDisplay.bounds,
                    workArea: primaryDisplay.workArea
                }
            };
        });
        
        // Handle file operations
        ipcMain.handle('show-save-dialog', async (event, options) => {
            const result = await dialog.showSaveDialog(this.mainWindow, options);
            return result;
        });
        
        ipcMain.handle('show-open-dialog', async (event, options) => {
            const result = await dialog.showOpenDialog(this.mainWindow, options);
            return result;
        });
        
        // Handle external links
        ipcMain.handle('open-external', (event, url) => {
            shell.openExternal(url);
        });
        
        // Handle app info
        ipcMain.handle('get-app-info', () => {
            return {
                version: app.getVersion(),
                name: app.getName(),
                path: app.getAppPath(),
                userData: app.getPath('userData'),
                platform: process.platform,
                arch: process.arch
            };
        });
        
        console.log('IPC handlers setup complete');
    }
    
    setupMenu() {
        if (process.platform === 'darwin') {
            // macOS menu
            const template = [
                {
                    label: app.getName(),
                    submenu: [
                        { role: 'about' },
                        { type: 'separator' },
                        { role: 'services' },
                        { type: 'separator' },
                        { role: 'hide' },
                        { role: 'hideothers' },
                        { role: 'unhide' },
                        { type: 'separator' },
                        { role: 'quit' }
                    ]
                },
                {
                    label: 'Edit',
                    submenu: [
                        { role: 'undo' },
                        { role: 'redo' },
                        { type: 'separator' },
                        { role: 'cut' },
                        { role: 'copy' },
                        { role: 'paste' },
                        { role: 'selectall' }
                    ]
                },
                {
                    label: 'View',
                    submenu: [
                        { role: 'reload' },
                        { role: 'forcereload' },
                        { role: 'toggledevtools' },
                        { type: 'separator' },
                        { role: 'resetzoom' },
                        { role: 'zoomin' },
                        { role: 'zoomout' },
                        { type: 'separator' },
                        { role: 'togglefullscreen' }
                    ]
                },
                {
                    label: 'Window',
                    submenu: [
                        { role: 'minimize' },
                        { role: 'close' }
                    ]
                }
            ];
            
            const menu = Menu.buildFromTemplate(template);
            Menu.setApplicationMenu(menu);
        } else {
            // Windows/Linux - no menu by default
            Menu.setApplicationMenu(null);
        }
    }
    
    toggleWindow() {
        if (!this.mainWindow) {
            this.createMainWindow();
            return;
        }
        
        if (this.mainWindow.isVisible()) {
            this.mainWindow.hide();
        } else {
            this.mainWindow.show();
            this.mainWindow.focus();
        }
    }
    
    openSettings() {
        if (this.mainWindow) {
            this.mainWindow.show();
            this.mainWindow.focus();
            this.mainWindow.webContents.send('open-settings');
        }
    }
    
    showAbout() {
        dialog.showMessageBox(this.mainWindow, {
            type: 'info',
            title: 'About TactiBird',
            message: 'TactiBird',
            detail: `Version: ${app.getVersion()}\nAI-powered coaching overlay for Teamfight Tactics\n\nBuilt with Electron and love for TFT.`,
            buttons: ['OK']
        });
    }
    
    quitApp() {
        this.isQuitting = true;
        app.quit();
    }
    
    // Window management utilities
    centerWindow() {
        if (!this.mainWindow) return;
        
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workAreaSize;
        const windowBounds = this.mainWindow.getBounds();
        
        const x = Math.round((width - windowBounds.width) / 2);
        const y = Math.round((height - windowBounds.height) / 2);
        
        this.mainWindow.setPosition(x, y);
    }
    
    snapToEdge() {
        if (!this.mainWindow) return;
        
        const primaryDisplay = screen.getPrimaryDisplay();
        const { width, height } = primaryDisplay.workArea;
        const windowBounds = this.mainWindow.getBounds();
        
        // Snap to right edge by default
        const x = width - windowBounds.width - 20;
        const y = 50;
        
        this.mainWindow.setPosition(x, y);
    }
    
    // Transparency and visual effects
    setWindowClickThrough(clickThrough) {
        if (!this.mainWindow) return;
        
        this.mainWindow.setIgnoreMouseEvents(clickThrough, { forward: true });
    }
    
    // Auto-updater integration (if needed)
    setupAutoUpdater() {
        if (isDev) return;
        
        // Auto-updater setup would go here
        // const { autoUpdater } = require('electron-updater');
        // autoUpdater.checkForUpdatesAndNotify();
    }
    
    // Error handling
    setupErrorHandling() {
        process.on('uncaughtException', (error) => {
            console.error('Uncaught Exception:', error);
            
            dialog.showErrorBox('Unexpected Error', 
                `An unexpected error occurred:\n\n${error.message}\n\nThe application will continue running, but you may want to restart it.`
            );
        });
        
        process.on('unhandledRejection', (reason, promise) => {
            console.error('Unhandled Rejection at:', promise, 'reason:', reason);
        });
    }
    
    // Development helpers
    setupDevelopmentHelpers() {
        if (!isDev) return;
        
        // Reload on changes
        try {
            require('electron-reload')(__dirname, {
                electron: require('electron'),
                hardResetMethod: 'exit'
            });
        } catch (e) {
            console.log('electron-reload not available');
        }
        
        // Install DevTools extensions
        const { default: installExtension, REACT_DEVELOPER_TOOLS } = require('electron-devtools-installer');
        
        app.whenReady().then(() => {
            installExtension(REACT_DEVELOPER_TOOLS)
                .then((name) => console.log(`Added Extension: ${name}`))
                .catch((err) => console.log('An error occurred: ', err));
        });
    }
}

// Initialize the application
new TactibirdElectronApp();