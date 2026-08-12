/**
 * TexturedViewer - VTK viewer for textured GLTF/GLB meshes
 *
 * Specialized for displaying textured 3D models with proper:
 * - Material handling (metallic=0, roughness=1)
 * - Texture extraction and application
 * - Two-sided lighting
 * - Closer camera distance for better texture viewing
 */

import { BaseViewer } from '../core/BaseViewer.js';
import { SettingTypes } from '../ui/SettingsPanel.js';
import { StandardControls } from '../ui/ControlsBar.js';

/**
 * Textured viewer configuration
 */
const TEXTURED_CONFIG = {
    // Camera - closer for texture detail
    cameraDistanceMultiplier: 2.0,

    // Lighting
    twoSidedLighting: true,

    // Features
    enableFields: false,
    enableTextures: true,
    enableScreenshot: true,
    enableAxisIndicator: false,
    enableSettings: true,
    enableControls: true,

    // Appearance
    backgroundColor: [0.15, 0.15, 0.15],
    defaultMeshColor: [1.0, 1.0, 1.0],
    showEdges: false
};

export class TexturedViewer extends BaseViewer {
    /**
     * Create a TexturedViewer
     * @param {HTMLElement} container - Container element
     * @param {Object} config - Additional configuration
     */
    constructor(container, config = {}) {
        // Merge with textured config
        const mergedConfig = {
            ...TEXTURED_CONFIG,
            ...config,
            // Settings panel configuration
            settingsConfig: {
                title: 'Settings',
                fields: [
                    {
                        type: SettingTypes.COLOR,
                        id: 'backgroundColor',
                        label: 'Background',
                        defaultValue: config.backgroundColor || TEXTURED_CONFIG.backgroundColor
                    },
                    {
                        type: SettingTypes.CHECKBOX,
                        id: 'showEdges',
                        label: 'Show Edges',
                        defaultValue: false
                    }
                ]
            },
            // Controls bar configuration
            controlsConfig: {
                controls: [
                    StandardControls.CAMERA_VIEWS,
                    StandardControls.SCREENSHOT,
                    StandardControls.SETTINGS
                ]
            }
        };

        super(container, mergedConfig);

        // Texture state
        this.textureInfo = null;
    }

    /**
     * Configure loaded actors for textured display
     * @private
     * @override
     */
    _configureLoadedActors(result) {
        if (this.textureManager && result.actors.length > 0) {
            // Apply texture fixes to all actors
            this.textureInfo = this.textureManager.applyTextureFixesMultiple(result.actors);

            // Configure renderer for texture display
            this.textureManager.configureRenderer(this.renderer);

            console.log('[TexturedViewer] Texture info:', this.textureInfo);

            // If no textures found, apply default white color
            if (!this.textureInfo.hasTexture) {
                result.actors.forEach(actor => {
                    this.textureManager.setActorColor(actor, this.config.defaultMeshColor);
                });
            }
        }
    }

    /**
     * Update info overlay with texture information
     * @private
     * @override
     */
    _updateInfoOverlay(result) {
        if (!this.infoOverlay) return;

        const lines = [`File: ${this.currentFilename}`];

        if (result.actors) {
            lines.push(`Actors: ${result.actors.length}`);
        }

        if (this.textureInfo) {
            lines.push(`Textured: ${this.textureInfo.hasTexture ? 'Yes' : 'No'}`);
            if (this.textureInfo.texturedActors > 0) {
                lines.push(`Textured actors: ${this.textureInfo.texturedActors}`);
            }
        }

        this.infoOverlay.innerHTML = lines.join('<br>');
    }

    /**
     * Get texture information
     * @returns {Object|null}
     */
    getTextureInfo() {
        return this.textureInfo;
    }

    /**
     * Check if current mesh has textures
     * @returns {boolean}
     */
    hasTextures() {
        return this.textureInfo?.hasTexture || false;
    }
}

/**
 * Create a TexturedViewer instance
 * @param {HTMLElement} container
 * @param {Object} config
 * @returns {TexturedViewer}
 */
export function createTexturedViewer(container, config = {}) {
    const viewer = new TexturedViewer(container, config);
    viewer.initialize();
    return viewer;
}

export default TexturedViewer;
