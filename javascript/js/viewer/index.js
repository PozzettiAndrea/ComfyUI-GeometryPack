/**
 * GeomPack VTK Viewer - Main Entry Point
 *
 * This module exports the ViewerFactory and all viewer components
 * for creating unified VTK viewers.
 */

// Utils
import ColorUtils from './utils/ColorUtils.js';
import BoundsUtils from './utils/BoundsUtils.js';
import MessageHandler from './utils/MessageHandler.js';

// Core
import CameraController from './core/CameraController.js';
import ActorManager, { Representation } from './core/ActorManager.js';
import BaseViewer from './core/BaseViewer.js';

// UI
import SettingsPanel, { SettingTypes } from './ui/SettingsPanel.js';
import ControlsBar, { ControlTypes, StandardControls } from './ui/ControlsBar.js';

// Loaders
import LoaderFactory, { FormatTypes } from './loaders/LoaderFactory.js';
import STLLoader from './loaders/STLLoader.js';
import OBJLoader from './loaders/OBJLoader.js';
import VTPLoader from './loaders/VTPLoader.js';
import GLTFLoader from './loaders/GLTFLoader.js';

// Features
import FieldVisualization from './features/FieldVisualization.js';
import TextureManager from './features/TextureManager.js';
import ScreenshotCapture from './features/ScreenshotCapture.js';
import AxisIndicator from './features/AxisIndicator.js';

// Viewers
import StandardViewer, { createStandardViewer } from './viewers/StandardViewer.js';
import TexturedViewer, { createTexturedViewer } from './viewers/TexturedViewer.js';
import DualViewer, { createDualViewer, DualLayoutMode } from './viewers/DualViewer.js';

// Re-export utilities
export { ColorUtils, BoundsUtils, MessageHandler };

// Re-export core
export { CameraController, ActorManager, Representation, BaseViewer };

// Re-export UI
export { SettingsPanel, SettingTypes, ControlsBar, ControlTypes, StandardControls };

// Re-export loaders
export { LoaderFactory, FormatTypes, STLLoader, OBJLoader, VTPLoader, GLTFLoader };

// Re-export features
export { FieldVisualization, TextureManager, ScreenshotCapture, AxisIndicator };

// Re-export viewers
export { StandardViewer, TexturedViewer, DualViewer, DualLayoutMode };
export { createStandardViewer, createTexturedViewer, createDualViewer };

// Export individual utility functions for convenience
export { rgbToString, hexToRgb, rgbToHex } from './utils/ColorUtils.js';
export { getCenter, getMaxDimension, getDimensions, combineBounds, isValidBounds } from './utils/BoundsUtils.js';
export {
    MessageTypes,
    sendToParent,
    sendError,
    sendScreenshot,
    sendMeshLoaded,
    createMessageListener,
    extractFilename
} from './utils/MessageHandler.js';

// Export loader factory functions
export {
    loadMesh,
    loadMeshes,
    detectFormat,
    requiresRenderer,
    supportsTextures,
    supportsFields,
    getLoaderConfig
} from './loaders/LoaderFactory.js';

/**
 * Viewer types for ViewerFactory
 */
export const ViewerTypes = {
    STANDARD: 'standard',
    TEXTURED: 'textured',
    DUAL: 'dual'
};

/**
 * ViewerFactory - Creates viewer instances
 *
 * Usage:
 *   const viewer = ViewerFactory.create('standard', containerEl);
 *   await viewer.initialize();
 *   await viewer.loadMesh('/path/to/mesh.stl');
 */
export const ViewerFactory = {
    /**
     * Create a viewer instance
     * @param {string} type - Viewer type ('standard', 'textured', 'dual')
     * @param {HTMLElement|HTMLElement[]} container - Container element(s)
     * @param {Object} options - Viewer options
     * @returns {BaseViewer|DualViewer} Viewer instance
     */
    create(type, container, options = {}) {
        console.log(`[GeomPack Viewer] Creating ${type} viewer`);

        switch (type) {
            case ViewerTypes.STANDARD:
            case 'standard':
                return new StandardViewer(container, options);

            case ViewerTypes.TEXTURED:
            case 'textured':
                return new TexturedViewer(container, options);

            case ViewerTypes.DUAL:
            case 'dual':
                // Dual viewer expects two containers
                if (Array.isArray(container)) {
                    return new DualViewer(container[0], container[1], options);
                } else if (options.containerB) {
                    return new DualViewer(container, options.containerB, options);
                } else {
                    throw new Error('DualViewer requires two containers');
                }

            default:
                console.warn(`[GeomPack Viewer] Unknown viewer type: ${type}, falling back to standard`);
                return new StandardViewer(container, options);
        }
    },

    /**
     * Create and initialize a viewer
     * @param {string} type - Viewer type
     * @param {HTMLElement|HTMLElement[]} container - Container element(s)
     * @param {Object} options - Viewer options
     * @returns {Promise<BaseViewer|DualViewer>} Initialized viewer
     */
    async createAndInit(type, container, options = {}) {
        const viewer = this.create(type, container, options);
        await viewer.initialize();
        return viewer;
    }
};

// Default export
export default {
    // Utils
    ColorUtils,
    BoundsUtils,
    MessageHandler,

    // Core
    CameraController,
    ActorManager,
    Representation,
    BaseViewer,

    // UI
    SettingsPanel,
    SettingTypes,
    ControlsBar,
    ControlTypes,
    StandardControls,

    // Loaders
    LoaderFactory,
    FormatTypes,
    STLLoader,
    OBJLoader,
    VTPLoader,
    GLTFLoader,

    // Features
    FieldVisualization,
    TextureManager,
    ScreenshotCapture,
    AxisIndicator,

    // Viewers
    StandardViewer,
    TexturedViewer,
    DualViewer,
    DualLayoutMode,
    ViewerTypes,

    // Factory functions
    createStandardViewer,
    createTexturedViewer,
    createDualViewer,

    // Factory
    ViewerFactory
};
