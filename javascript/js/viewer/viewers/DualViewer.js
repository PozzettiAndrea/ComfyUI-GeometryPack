/**
 * DualViewer - VTK viewer for dual mesh comparison
 *
 * Supports side-by-side and overlay layouts with:
 * - Camera synchronization between viewports
 * - Synchronized scalar field ranges
 * - Multiple layout modes (side-by-side, overlay, split)
 * - Independent and linked settings
 */

import { CameraController } from '../core/CameraController.js';
import { ActorManager } from '../core/ActorManager.js';
import { loadMesh, supportsTextures } from '../loaders/LoaderFactory.js';
import { FieldVisualization } from '../features/FieldVisualization.js';
import { TextureManager } from '../features/TextureManager.js';
import { ScreenshotCapture } from '../features/ScreenshotCapture.js';
import { sendMeshLoaded, sendError, createMessageListener, extractFilename } from '../utils/MessageHandler.js';
import { combineBounds, isValidBounds } from '../utils/BoundsUtils.js';

/**
 * Layout modes for dual viewer
 */
export const LayoutMode = {
    SIDE_BY_SIDE: 'side-by-side',
    OVERLAY: 'overlay',
    VERTICAL: 'vertical'
};

/**
 * Dual viewer configuration
 */
const DUAL_CONFIG = {
    // Camera
    cameraDistanceMultiplier: 2.5,

    // Layout
    layout: LayoutMode.SIDE_BY_SIDE,

    // Sync options
    syncCameras: true,
    syncFields: true,

    // Appearance
    backgroundColor: [0.15, 0.15, 0.15],
    defaultMeshColorA: [0.6, 0.8, 1.0],
    defaultMeshColorB: [0.6, 0.8, 1.0],
    showEdges: false,

    // Features
    enableFields: true,
    enableTextures: true
};

export class DualViewer {
    /**
     * Create a DualViewer
     * @param {HTMLElement} containerA - Container for first viewport
     * @param {HTMLElement} containerB - Container for second viewport
     * @param {Object} config - Configuration options
     */
    constructor(containerA, containerB, config = {}) {
        this.containerA = containerA;
        this.containerB = containerB;
        this.config = { ...DUAL_CONFIG, ...config };

        // VTK objects
        this.vtk = null;

        // Viewports (each has its own renderer, render window, etc.)
        this.viewportA = null;
        this.viewportB = null;

        // Feature modules (shared)
        this.actorManager = null;
        this.fieldVisualization = null;

        // State
        this.layout = this.config.layout;
        this.syncCameras = this.config.syncCameras;
        this.syncFields = this.config.syncFields;
        this.isInitialized = false;

        // Camera sync state
        this._cameraObserver = null;
        this._isUpdatingCamera = false;

        // Message listener cleanup
        this._messageCleanup = null;
    }

    /**
     * Initialize the dual viewer
     */
    async initialize() {
        if (this.isInitialized) return;

        this.vtk = window.vtk;
        if (!this.vtk) {
            throw new Error('VTK.js not loaded');
        }

        // Create both viewports
        this.viewportA = this._createViewport(this.containerA, 'A');
        this.viewportB = this._createViewport(this.containerB, 'B');

        // Initialize shared modules
        this.actorManager = new ActorManager(this.vtk, {
            defaultColor: this.config.defaultMeshColorA
        });
        this.fieldVisualization = new FieldVisualization(this.vtk, this.viewportA.renderWindow);

        // Setup camera sync if enabled
        if (this.syncCameras) {
            this._setupCameraSync();
        }

        // Setup resize handlers
        this._setupResizeHandlers();

        // Setup message listener
        this._setupMessageListener();

        this.isInitialized = true;
        console.log('[DualViewer] Initialized');
    }

    /**
     * Create a viewport
     * @private
     * @param {HTMLElement} container
     * @param {string} id
     * @returns {Object}
     */
    _createViewport(container, id) {
        const renderWindow = this.vtk.Rendering.Core.vtkRenderWindow.newInstance();
        const renderer = this.vtk.Rendering.Core.vtkRenderer.newInstance();
        renderer.setBackground(...this.config.backgroundColor);
        renderWindow.addRenderer(renderer);

        const openGLRenderWindow = this.vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
        openGLRenderWindow.setContainer(container);
        renderWindow.addView(openGLRenderWindow);

        const interactor = this.vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
        interactor.setView(openGLRenderWindow);
        interactor.initialize();
        interactor.bindEvents(container);

        const interactorStyle = this.vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
        interactor.setInteractorStyle(interactorStyle);

        const { width, height } = container.getBoundingClientRect();
        openGLRenderWindow.setSize(width, height);

        const cameraController = new CameraController(renderer, {
            distanceMultiplier: this.config.cameraDistanceMultiplier
        });

        // Create texture manager for this viewport
        const textureManager = new TextureManager(this.vtk, renderWindow);

        return {
            id,
            container,
            renderWindow,
            renderer,
            openGLRenderWindow,
            interactor,
            interactorStyle,
            cameraController,
            textureManager,
            actors: [],
            polyData: null,
            filename: null
        };
    }

    /**
     * Setup camera synchronization between viewports
     * @private
     */
    _setupCameraSync() {
        const syncCamera = (sourceViewport, targetViewport) => {
            if (this._isUpdatingCamera) return;
            this._isUpdatingCamera = true;

            const sourceCamera = sourceViewport.renderer.getActiveCamera();
            const targetCamera = targetViewport.renderer.getActiveCamera();

            targetCamera.setPosition(...sourceCamera.getPosition());
            targetCamera.setFocalPoint(...sourceCamera.getFocalPoint());
            targetCamera.setViewUp(...sourceCamera.getViewUp());
            targetCamera.setParallelScale(sourceCamera.getParallelScale());

            targetViewport.renderWindow.render();
            this._isUpdatingCamera = false;
        };

        // Subscribe to camera modified events
        const cameraA = this.viewportA.renderer.getActiveCamera();
        const cameraB = this.viewportB.renderer.getActiveCamera();

        cameraA.onModified(() => {
            if (this.syncCameras) {
                syncCamera(this.viewportA, this.viewportB);
            }
        });

        cameraB.onModified(() => {
            if (this.syncCameras) {
                syncCamera(this.viewportB, this.viewportA);
            }
        });
    }

    /**
     * Setup resize handlers for both viewports
     * @private
     */
    _setupResizeHandlers() {
        const handleResize = (viewport) => {
            const { width, height } = viewport.container.getBoundingClientRect();
            if (width > 0 && height > 0) {
                viewport.openGLRenderWindow.setSize(width, height);
                viewport.renderWindow.render();
            }
        };

        this._resizeObserverA = new ResizeObserver(() => handleResize(this.viewportA));
        this._resizeObserverB = new ResizeObserver(() => handleResize(this.viewportB));

        this._resizeObserverA.observe(this.viewportA.container);
        this._resizeObserverB.observe(this.viewportB.container);
    }

    /**
     * Setup message listener
     * @private
     */
    _setupMessageListener() {
        this._messageCleanup = createMessageListener((data) => {
            switch (data.type) {
                case 'loadMeshA':
                    this.loadMesh('A', data.url);
                    break;
                case 'loadMeshB':
                    this.loadMesh('B', data.url);
                    break;
                case 'loadMesh':
                    // Load to both or specified viewport
                    if (data.viewport) {
                        this.loadMesh(data.viewport, data.url);
                    } else {
                        this.loadMesh('A', data.url);
                    }
                    break;
                case 'setLayout':
                    this.setLayout(data.layout);
                    break;
                case 'setCameraSync':
                    this.setCameraSync(data.enabled);
                    break;
                case 'setField':
                    this.applyField(data.field, data.options);
                    break;
                case 'screenshot':
                    this.captureScreenshot();
                    break;
            }
        });
    }

    /**
     * Load mesh into a viewport
     * @param {string} viewport - 'A' or 'B'
     * @param {string} url - Mesh URL
     * @returns {Promise<Object>}
     */
    async loadMesh(viewport, url) {
        const vp = viewport === 'A' ? this.viewportA : this.viewportB;
        const defaultColor = viewport === 'A'
            ? this.config.defaultMeshColorA
            : this.config.defaultMeshColorB;

        try {
            // Clear existing actors
            vp.actors.forEach(actor => vp.renderer.removeActor(actor));
            vp.actors = [];

            // Load mesh
            const result = await loadMesh(url, this.vtk, {
                renderer: vp.renderer
            });

            vp.actors = result.actors;
            vp.polyData = result.polyData;
            vp.filename = extractFilename(url);

            // Add actors to renderer
            result.actors.forEach(actor => {
                vp.renderer.addActor(actor);
            });

            // Configure actors based on format
            if (result.hasTexture && this.config.enableTextures) {
                // Apply texture fixes for GLTF/GLB
                vp.textureManager.applyTextureFixesMultiple(result.actors);
                vp.textureManager.configureRenderer(vp.renderer);
            } else if (result.hasVertexColors) {
                // Mesh has vertex colors - don't override with default color
                // Just configure edges if needed, colors are already set up by loader
                result.actors.forEach(actor => {
                    const property = actor.getProperty();
                    if (property && this.config.showEdges) {
                        property.setEdgeVisibility(true);
                    }
                });
            } else {
                // Standard actor configuration - apply default color
                this.actorManager.configureActors(result.actors, {
                    color: defaultColor,
                    showEdges: this.config.showEdges
                });
            }

            // Position camera
            this._positionCameraForViewport(vp);

            // Render
            vp.renderWindow.render();

            // Notify parent
            sendMeshLoaded({
                viewport,
                filename: vp.filename,
                format: result.format,
                hasTexture: result.hasTexture || false,
                hasVertexColors: result.hasVertexColors || false
            });

            return result;

        } catch (error) {
            console.error(`[DualViewer] Error loading mesh ${viewport}:`, error);
            sendError(`Failed to load mesh ${viewport}: ${error.message}`);
            throw error;
        }
    }

    /**
     * Position camera for a viewport
     * @private
     */
    _positionCameraForViewport(viewport) {
        if (viewport.actors.length === 0) return;

        let bounds = viewport.actors[0].getBounds();
        for (let i = 1; i < viewport.actors.length; i++) {
            bounds = combineBounds(bounds, viewport.actors[i].getBounds());
        }

        if (isValidBounds(bounds)) {
            viewport.cameraController.positionInitialCamera(bounds);
        }
    }

    /**
     * Apply field visualization to both viewports
     * @param {string} fieldName - Field name
     * @param {Object} options - Options including colormap, range
     */
    applyField(fieldName, options = {}) {
        if (!this.config.enableFields) return;

        // Calculate synchronized range if enabled
        let range = options.range;
        if (this.syncFields && !range) {
            const polydatas = [
                this.viewportA.polyData,
                this.viewportB.polyData
            ].filter(Boolean);

            if (polydatas.length > 0) {
                range = this.fieldVisualization.calculateSynchronizedRange(polydatas, fieldName);
            }
        }

        // Apply to viewport A
        if (this.viewportA.polyData && this.viewportA.actors.length > 0) {
            const mapperA = this.viewportA.actors[0].getMapper();
            this.fieldVisualization.applyField(mapperA, this.viewportA.polyData, fieldName, {
                ...options,
                range,
                render: false
            });
        }

        // Apply to viewport B
        if (this.viewportB.polyData && this.viewportB.actors.length > 0) {
            const mapperB = this.viewportB.actors[0].getMapper();
            // Need a separate FieldVisualization for viewport B's render window
            const fieldVizB = new FieldVisualization(this.vtk, this.viewportB.renderWindow);
            fieldVizB.applyField(mapperB, this.viewportB.polyData, fieldName, {
                ...options,
                range,
                render: false
            });
        }

        // Render both
        this.render();
    }

    /**
     * Set layout mode
     * @param {string} layout - LayoutMode value
     */
    setLayout(layout) {
        this.layout = layout;
        // Layout changes are handled by CSS classes on the containers
        // This method just updates the state; the parent should update CSS
        console.log(`[DualViewer] Layout set to: ${layout}`);
    }

    /**
     * Set camera sync enabled
     * @param {boolean} enabled
     */
    setCameraSync(enabled) {
        this.syncCameras = enabled;
        console.log(`[DualViewer] Camera sync: ${enabled}`);
    }

    /**
     * Set field range sync enabled
     * @param {boolean} enabled
     */
    setFieldSync(enabled) {
        this.syncFields = enabled;
    }

    /**
     * Set camera view for both viewports
     * @param {string} direction - '+X', '-X', etc.
     */
    setCameraView(direction) {
        // Get combined bounds from both viewports
        const allActors = [...this.viewportA.actors, ...this.viewportB.actors];
        if (allActors.length === 0) return;

        let bounds = allActors[0].getBounds();
        for (let i = 1; i < allActors.length; i++) {
            bounds = combineBounds(bounds, allActors[i].getBounds());
        }

        // Apply to viewport A (will sync to B if enabled)
        this.viewportA.cameraController.setCameraView(direction, bounds);

        // If sync is disabled, also update B
        if (!this.syncCameras) {
            this.viewportB.cameraController.setCameraView(direction, bounds);
        }

        this.render();
    }

    /**
     * Capture screenshot of both viewports
     * @returns {Promise<Object>}
     */
    async captureScreenshot() {
        const screenshotA = new ScreenshotCapture(this.viewportA.renderWindow);
        const screenshotB = new ScreenshotCapture(this.viewportB.renderWindow);

        const [imageA, imageB] = await Promise.all([
            screenshotA.capture(),
            screenshotB.capture()
        ]);

        // Send combined screenshot info
        window.parent?.postMessage({
            type: 'screenshot',
            data: {
                viewportA: imageA,
                viewportB: imageB
            }
        }, '*');

        return { viewportA: imageA, viewportB: imageB };
    }

    /**
     * Render both viewports
     */
    render() {
        this.viewportA.renderWindow.render();
        this.viewportB.renderWindow.render();
    }

    /**
     * Get viewport
     * @param {string} id - 'A' or 'B'
     * @returns {Object}
     */
    getViewport(id) {
        return id === 'A' ? this.viewportA : this.viewportB;
    }

    /**
     * Cleanup and destroy
     */
    destroy() {
        if (this._messageCleanup) {
            this._messageCleanup();
        }

        if (this._resizeObserverA) {
            this._resizeObserverA.disconnect();
        }
        if (this._resizeObserverB) {
            this._resizeObserverB.disconnect();
        }

        // Destroy viewports
        [this.viewportA, this.viewportB].forEach(vp => {
            if (!vp) return;
            vp.actors.forEach(actor => vp.renderer.removeActor(actor));
            vp.interactor.unbindEvents();
            vp.openGLRenderWindow.delete();
            vp.renderWindow.delete();
        });

        console.log('[DualViewer] Destroyed');
    }
}

/**
 * Create a DualViewer instance
 * @param {HTMLElement} containerA
 * @param {HTMLElement} containerB
 * @param {Object} config
 * @returns {DualViewer}
 */
export function createDualViewer(containerA, containerB, config = {}) {
    const viewer = new DualViewer(containerA, containerB, config);
    viewer.initialize();
    return viewer;
}

export { LayoutMode as DualLayoutMode };
export default DualViewer;
