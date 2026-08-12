/**
 * BaseViewer - Foundation class for VTK viewers
 *
 * Provides core lifecycle management, VTK setup, and feature composition.
 * Subclasses configure features and behavior through options.
 */

import { CameraController } from './CameraController.js';
import { ActorManager, Representation } from './ActorManager.js';
import { SettingsPanel } from '../ui/SettingsPanel.js';
import { ControlsBar, StandardControls } from '../ui/ControlsBar.js';
import { loadMesh } from '../loaders/LoaderFactory.js';
import { ScreenshotCapture } from '../features/ScreenshotCapture.js';
import { AxisIndicator } from '../features/AxisIndicator.js';
import { FieldVisualization } from '../features/FieldVisualization.js';
import { TextureManager } from '../features/TextureManager.js';
import { sendMeshLoaded, sendError, createMessageListener, extractFilename } from '../utils/MessageHandler.js';
import { getCenter, getMaxDimension, isValidBounds } from '../utils/BoundsUtils.js';

/**
 * Default viewer configuration
 */
const DEFAULT_CONFIG = {
    // Appearance
    backgroundColor: [0.15, 0.15, 0.15],
    defaultMeshColor: [0.6, 0.8, 1.0],
    edgeColor: [0.2, 0.2, 0.2],
    edgeWidth: 1.0,
    showEdges: false,

    // Camera
    cameraDistanceMultiplier: 2.5,
    parallelProjection: false,

    // Lighting
    twoSidedLighting: false,

    // Features (enabled/disabled)
    enableFields: true,
    enableTextures: false,
    enableScreenshot: true,
    enableAxisIndicator: false,
    enableSettings: true,
    enableControls: true,

    // UI
    showLoadingOverlay: true,
    showInfoOverlay: true
};

export class BaseViewer {
    /**
     * Create a BaseViewer
     * @param {HTMLElement} container - Container element for the viewer
     * @param {Object} config - Configuration options
     */
    constructor(container, config = {}) {
        this.container = container;
        this.config = { ...DEFAULT_CONFIG, ...config };

        // VTK objects (initialized in initialize())
        this.vtk = null;
        this.renderWindow = null;
        this.renderer = null;
        this.interactor = null;
        this.openGLRenderWindow = null;

        // Feature modules
        this.cameraController = null;
        this.actorManager = null;
        this.settingsPanel = null;
        this.controlsBar = null;
        this.screenshotCapture = null;
        this.axisIndicator = null;
        this.fieldVisualization = null;
        this.textureManager = null;

        // State
        this.actors = [];
        this.polyDatas = [];
        this.currentFilename = null;
        this.isInitialized = false;

        // UI elements
        this.loadingOverlay = null;
        this.infoOverlay = null;

        // Message listener cleanup
        this._messageCleanup = null;
    }

    /**
     * Initialize the viewer
     * Must be called before loading meshes
     */
    async initialize() {
        if (this.isInitialized) {
            console.warn('[BaseViewer] Already initialized');
            return;
        }

        // Get VTK from global
        this.vtk = window.vtk;
        if (!this.vtk) {
            throw new Error('VTK.js not loaded. Include vtk-gltf.js before initializing viewer.');
        }

        // Create VTK rendering pipeline
        this._createRenderingPipeline();

        // Initialize feature modules
        this._initializeFeatures();

        // Create UI elements
        this._createUI();

        // Setup resize handling
        this._setupResizeHandler();

        // Setup message listener for loading meshes
        this._setupMessageListener();

        this.isInitialized = true;
        console.log('[BaseViewer] Initialized');
    }

    /**
     * Create VTK rendering pipeline
     * @private
     */
    _createRenderingPipeline() {
        // Create render window
        this.renderWindow = this.vtk.Rendering.Core.vtkRenderWindow.newInstance();

        // Create renderer
        this.renderer = this.vtk.Rendering.Core.vtkRenderer.newInstance();
        this.renderer.setBackground(...this.config.backgroundColor);
        this.renderWindow.addRenderer(this.renderer);

        // Two-sided lighting
        if (this.config.twoSidedLighting) {
            this.renderer.setTwoSidedLighting(true);
        }

        // Create OpenGL render window
        this.openGLRenderWindow = this.vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
        this.openGLRenderWindow.setContainer(this.container);
        this.renderWindow.addView(this.openGLRenderWindow);

        // Create interactor
        this.interactor = this.vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
        this.interactor.setView(this.openGLRenderWindow);
        this.interactor.initialize();
        this.interactor.bindEvents(this.container);

        // Set interactor style
        const interactorStyle = this.vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
        this.interactor.setInteractorStyle(interactorStyle);

        // Size to container
        const { width, height } = this.container.getBoundingClientRect();
        this.openGLRenderWindow.setSize(width, height);
    }

    /**
     * Initialize feature modules
     * @private
     */
    _initializeFeatures() {
        // Camera controller
        this.cameraController = new CameraController(this.renderer, {
            distanceMultiplier: this.config.cameraDistanceMultiplier,
            parallelProjection: this.config.parallelProjection
        });

        // Actor manager
        this.actorManager = new ActorManager(this.vtk, {
            defaultColor: this.config.defaultMeshColor,
            edgeColor: this.config.edgeColor,
            edgeWidth: this.config.edgeWidth,
            showEdges: this.config.showEdges
        });

        // Screenshot capture
        if (this.config.enableScreenshot) {
            this.screenshotCapture = new ScreenshotCapture(this.renderWindow);
        }

        // Axis indicator
        if (this.config.enableAxisIndicator) {
            this.axisIndicator = new AxisIndicator(this.vtk, this.interactor, {
                enabled: true
            });
        }

        // Field visualization
        if (this.config.enableFields) {
            this.fieldVisualization = new FieldVisualization(this.vtk, this.renderWindow);
        }

        // Texture manager
        if (this.config.enableTextures) {
            this.textureManager = new TextureManager(this.vtk, this.renderWindow);
        }
    }

    /**
     * Create UI elements
     * @private
     */
    _createUI() {
        // Loading overlay
        if (this.config.showLoadingOverlay) {
            this.loadingOverlay = this._createLoadingOverlay();
        }

        // Info overlay
        if (this.config.showInfoOverlay) {
            this.infoOverlay = this._createInfoOverlay();
        }

        // Settings panel
        if (this.config.enableSettings && this.config.settingsConfig) {
            this.settingsPanel = new SettingsPanel(this.config.settingsConfig);
            this.settingsPanel.onApply((settings) => this._onSettingsApply(settings));
        }

        // Controls bar
        if (this.config.enableControls && this.config.controlsConfig) {
            this.controlsBar = new ControlsBar(this.config.controlsConfig);
            this._setupControlsHandlers();
        }
    }

    /**
     * Create loading overlay element
     * @private
     * @returns {HTMLElement}
     */
    _createLoadingOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'loading';
        overlay.className = 'loading';
        overlay.innerHTML = `
            <div class="spinner"></div>
            <div class="loading-text">Loading mesh...</div>
            <div class="loading-progress" style="display:none;width:140px;height:6px;margin-top:8px;
                 background:rgba(255,255,255,0.18);border-radius:3px;overflow:hidden;">
                <div class="loading-progress-bar" style="width:0%;height:100%;
                     background:#00c8ff;transition:width 0.1s linear;"></div>
            </div>
        `;
        this.container.parentElement.appendChild(overlay);
        return overlay;
    }

    /**
     * Update the download progress bar.
     * @param {number|null} fraction - 0..1, or null for indeterminate (hides the bar)
     */
    setProgress(fraction) {
        if (!this.loadingOverlay) return;
        const wrap = this.loadingOverlay.querySelector('.loading-progress');
        const bar = this.loadingOverlay.querySelector('.loading-progress-bar');
        const textEl = this.loadingOverlay.querySelector('.loading-text');
        if (!wrap || !bar) return;
        if (fraction == null || isNaN(fraction)) {
            wrap.style.display = 'none';        // indeterminate -> spinner only
            return;
        }
        const pct = Math.max(0, Math.min(100, Math.round(fraction * 100)));
        wrap.style.display = 'block';
        bar.style.width = pct + '%';
        if (textEl) textEl.textContent = `Downloading mesh… ${pct}%`;
    }

    /**
     * Create info overlay element
     * @private
     * @returns {HTMLElement}
     */
    _createInfoOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'meshInfo';
        overlay.className = 'mesh-info';
        this.container.parentElement.appendChild(overlay);
        return overlay;
    }

    /**
     * Setup resize handler
     * @private
     */
    _setupResizeHandler() {
        const resizeObserver = new ResizeObserver(() => {
            const { width, height } = this.container.getBoundingClientRect();
            if (width > 0 && height > 0) {
                this.openGLRenderWindow.setSize(width, height);
                this.renderWindow.render();
            }
        });
        resizeObserver.observe(this.container);
        this._resizeObserver = resizeObserver;
    }

    /**
     * Setup message listener for postMessage API
     * @private
     */
    _setupMessageListener() {
        this._messageCleanup = createMessageListener((data) => {
            if (data.type === 'loadMesh' && data.url) {
                this.loadMesh(data.url);
            } else if (data.type === 'screenshot') {
                this.captureScreenshot();
            }
        });
    }

    /**
     * Setup controls bar handlers
     * @private
     */
    _setupControlsHandlers() {
        if (!this.controlsBar) return;

        // Camera view buttons
        this.controlsBar.onAction('camera-view', (direction) => {
            this.setCameraView(direction);
        });

        // Screenshot button
        this.controlsBar.onAction('screenshot', () => {
            this.captureScreenshot();
        });

        // Settings button
        this.controlsBar.onAction('settings', () => {
            if (this.settingsPanel) {
                this.settingsPanel.open();
            }
        });

        // Edges toggle
        this.controlsBar.onAction('edges-toggle', (show) => {
            this.setEdgeVisibility(show);
        });
    }

    /**
     * Handle settings panel apply
     * @private
     * @param {Object} settings - Applied settings
     */
    _onSettingsApply(settings) {
        // Apply to actor manager
        if (this.actorManager) {
            this.actorManager.applySettings(this.actors, settings);
        }

        // Apply background color
        if (settings.backgroundColor) {
            this.renderer.setBackground(...settings.backgroundColor);
        }

        this.render();
    }

    /**
     * Load a mesh from URL
     * @param {string} url - Mesh file URL
     * @returns {Promise<Object>} Load result with polyData, actors, fields
     */
    async loadMesh(url) {
        this.showLoading(true, 'Downloading mesh…');
        this.setProgress(0);

        try {
            // Clear existing actors
            this.clearScene();

            // Detect format and load (with download progress)
            const result = await loadMesh(url, this.vtk, {
                renderer: this.renderer,
                onProgress: (frac) => this.setProgress(frac)
            });

            // Store state
            this.actors = result.actors;
            this.polyDatas = result.polyData ? [result.polyData] : [];
            this.currentFilename = extractFilename(url);

            // Add actors to renderer
            result.actors.forEach(actor => {
                this.renderer.addActor(actor);
            });

            // Configure actors
            this._configureLoadedActors(result);

            // Setup camera
            this._setupCameraForMesh();

            // Update UI
            this._updateInfoOverlay(result);
            this.showLoading(false);

            // Notify parent
            sendMeshLoaded({
                filename: this.currentFilename,
                format: result.format,
                actors: result.actors.length,
                hasTexture: result.hasTexture || false,
                hasVertexColors: result.hasVertexColors || false,
                fields: result.fields || []
            });

            this.render();
            return result;

        } catch (error) {
            console.error('[BaseViewer] Error loading mesh:', error);
            this.showLoading(false);
            sendError(error.message);
            throw error;
        }
    }

    /**
     * Configure actors after loading
     * @private
     * @param {Object} result - Load result
     */
    _configureLoadedActors(result) {
        // Apply texture fixes if enabled
        if (this.config.enableTextures && this.textureManager && result.hasTexture) {
            this.textureManager.applyTextureFixesMultiple(this.actors);
            this.textureManager.configureRenderer(this.renderer);
        } else if (result.hasVertexColors) {
            // Mesh has vertex colors - don't override with default color
            // Just configure edges if needed, colors are already set up by loader
            this.actors.forEach(actor => {
                const property = actor.getProperty();
                if (property) {
                    if (this.config.showEdges) {
                        property.setEdgeVisibility(true);
                        if (this.config.edgeColor) {
                            property.setEdgeColor(...this.config.edgeColor);
                        }
                    }
                }
            });
        } else {
            // Apply standard actor configuration
            this.actorManager.configureActors(this.actors, {
                color: this.config.defaultMeshColor,
                showEdges: this.config.showEdges,
                edgeColor: this.config.edgeColor
            });
        }
    }

    /**
     * Setup camera position for loaded mesh
     * @private
     */
    _setupCameraForMesh() {
        if (this.actors.length === 0) return;

        // Get combined bounds
        let bounds = this.actors[0].getBounds();
        for (let i = 1; i < this.actors.length; i++) {
            const actorBounds = this.actors[i].getBounds();
            bounds = [
                Math.min(bounds[0], actorBounds[0]),
                Math.max(bounds[1], actorBounds[1]),
                Math.min(bounds[2], actorBounds[2]),
                Math.max(bounds[3], actorBounds[3]),
                Math.min(bounds[4], actorBounds[4]),
                Math.max(bounds[5], actorBounds[5])
            ];
        }

        if (isValidBounds(bounds)) {
            this.cameraController.positionInitialCamera(bounds);
        }
    }

    /**
     * Update info overlay with mesh information
     * @private
     * @param {Object} result - Load result
     */
    _updateInfoOverlay(result) {
        if (!this.infoOverlay) return;

        const lines = [`File: ${this.currentFilename}`];

        if (result.polyData) {
            const numPoints = result.polyData.getNumberOfPoints();
            const numCells = result.polyData.getNumberOfCells();
            lines.push(`Points: ${numPoints.toLocaleString()}`);
            lines.push(`Cells: ${numCells.toLocaleString()}`);
        }

        if (result.hasTexture) {
            lines.push('Textured: Yes');
        }

        this.infoOverlay.innerHTML = lines.join('<br>');
    }

    /**
     * Clear the scene
     */
    clearScene() {
        this.actors.forEach(actor => {
            this.renderer.removeActor(actor);
        });
        this.actors = [];
        this.polyDatas = [];
    }

    /**
     * Set camera view direction
     * @param {string} direction - '+X', '-X', '+Y', '-Y', '+Z', '-Z'
     */
    setCameraView(direction) {
        if (this.actors.length === 0) return;

        const bounds = this.actors[0].getBounds();
        this.cameraController.setCameraView(direction, bounds);
        this.render();
    }

    /**
     * Set edge visibility
     * @param {boolean} visible
     */
    setEdgeVisibility(visible) {
        this.actors.forEach(actor => {
            this.actorManager.setEdgeVisibility(actor, visible);
        });
        this.render();
    }

    /**
     * Capture screenshot
     * @returns {Promise<string>} Base64 image data
     */
    async captureScreenshot() {
        if (this.screenshotCapture) {
            return this.screenshotCapture.captureAndSend();
        }
    }

    /**
     * Show/hide loading overlay
     * @param {boolean} show
     * @param {string} text - Optional loading text
     */
    showLoading(show, text = 'Loading mesh...') {
        if (this.loadingOverlay) {
            this.loadingOverlay.style.display = show ? 'flex' : 'none';
            const textEl = this.loadingOverlay.querySelector('.loading-text');
            if (textEl) {
                textEl.textContent = text;
            }
            if (!show) {
                // reset the progress bar for the next load
                const wrap = this.loadingOverlay.querySelector('.loading-progress');
                const bar = this.loadingOverlay.querySelector('.loading-progress-bar');
                if (wrap) wrap.style.display = 'none';
                if (bar) bar.style.width = '0%';
            }
        }
    }

    /**
     * Render the scene
     */
    render() {
        this.renderWindow.render();
    }

    /**
     * Resize the viewer
     * @param {number} width
     * @param {number} height
     */
    resize(width, height) {
        this.openGLRenderWindow.setSize(width, height);
        this.render();
    }

    /**
     * Get current actors
     * @returns {Object[]}
     */
    getActors() {
        return this.actors;
    }

    /**
     * Get renderer
     * @returns {Object}
     */
    getRenderer() {
        return this.renderer;
    }

    /**
     * Cleanup and destroy viewer
     */
    destroy() {
        // Cleanup message listener
        if (this._messageCleanup) {
            this._messageCleanup();
        }

        // Cleanup resize observer
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
        }

        // Cleanup axis indicator
        if (this.axisIndicator) {
            this.axisIndicator.destroy();
        }

        // Clear scene
        this.clearScene();

        // Delete VTK objects
        if (this.interactor) {
            this.interactor.unbindEvents();
        }
        if (this.openGLRenderWindow) {
            this.openGLRenderWindow.delete();
        }
        if (this.renderWindow) {
            this.renderWindow.delete();
        }

        // Remove UI elements
        if (this.loadingOverlay && this.loadingOverlay.parentElement) {
            this.loadingOverlay.parentElement.removeChild(this.loadingOverlay);
        }
        if (this.infoOverlay && this.infoOverlay.parentElement) {
            this.infoOverlay.parentElement.removeChild(this.infoOverlay);
        }

        console.log('[BaseViewer] Destroyed');
    }
}

export default BaseViewer;
