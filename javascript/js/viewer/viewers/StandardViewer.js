/**
 * StandardViewer - VTK viewer for standard mesh preview
 *
 * Supports STL, OBJ, VTP formats with:
 * - Scalar field visualization
 * - Colormap selection
 * - Settings panel (color, edges, lighting)
 * - Screenshot capture
 */

import { BaseViewer } from '../core/BaseViewer.js';
import { SettingTypes } from '../ui/SettingsPanel.js';
import { ControlTypes, StandardControls } from '../ui/ControlsBar.js';

/**
 * Standard viewer configuration
 */
const STANDARD_CONFIG = {
    // Camera
    cameraDistanceMultiplier: 2.5,

    // Features
    enableFields: true,
    enableTextures: false,
    enableScreenshot: true,
    enableAxisIndicator: false,
    enableSettings: true,
    enableControls: true,

    // Appearance
    backgroundColor: [0.15, 0.15, 0.15],
    defaultMeshColor: [0.6, 0.8, 1.0],
    showEdges: false
};

export class StandardViewer extends BaseViewer {
    /**
     * Create a StandardViewer
     * @param {HTMLElement} container - Container element
     * @param {Object} config - Additional configuration
     */
    constructor(container, config = {}) {
        // Merge with standard config
        const mergedConfig = {
            ...STANDARD_CONFIG,
            ...config,
            // Settings panel configuration
            settingsConfig: {
                title: 'Settings',
                fields: [
                    {
                        type: SettingTypes.COLOR,
                        id: 'meshColor',
                        label: 'Mesh Color',
                        defaultValue: config.defaultMeshColor || STANDARD_CONFIG.defaultMeshColor
                    },
                    {
                        type: SettingTypes.CHECKBOX,
                        id: 'showEdges',
                        label: 'Show Edges',
                        defaultValue: config.showEdges || STANDARD_CONFIG.showEdges
                    },
                    {
                        type: SettingTypes.COLOR,
                        id: 'edgeColor',
                        label: 'Edge Color',
                        defaultValue: config.edgeColor || [0.2, 0.2, 0.2]
                    },
                    {
                        type: SettingTypes.COLOR,
                        id: 'backgroundColor',
                        label: 'Background',
                        defaultValue: config.backgroundColor || STANDARD_CONFIG.backgroundColor
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

        // Field visualization state
        this.currentField = null;
        this.currentColormap = 'erdc_rainbow_bright';
        this.availableFields = [];
    }

    /**
     * Load mesh and extract field information
     * @param {string} url - Mesh file URL
     * @returns {Promise<Object>}
     */
    async loadMesh(url) {
        const result = await super.loadMesh(url);

        // Extract field information for VTP files
        if (this.fieldVisualization && this.polyDatas.length > 0) {
            const fieldInfo = this.fieldVisualization.extractFields(this.polyDatas[0]);
            this.availableFields = fieldInfo.allFields;

            // Notify parent about available fields
            if (this.availableFields.length > 0) {
                window.parent?.postMessage({
                    type: 'fieldsAvailable',
                    fields: this.availableFields.map(f => ({
                        name: f.fullName,
                        displayName: f.displayName,
                        type: f.type,
                        range: f.range
                    }))
                }, '*');
            }
        }

        return result;
    }

    /**
     * Apply scalar field visualization
     * @param {string} fieldName - Field name (e.g., 'point:temperature')
     * @param {Object} options - Visualization options
     */
    applyField(fieldName, options = {}) {
        if (!this.fieldVisualization || this.polyDatas.length === 0) {
            console.warn('[StandardViewer] Field visualization not available');
            return;
        }

        const mapper = this.actors[0]?.getMapper();
        if (!mapper) return;

        this.fieldVisualization.applyField(
            mapper,
            this.polyDatas[0],
            fieldName,
            {
                colormap: options.colormap || this.currentColormap,
                range: options.range,
                render: true
            }
        );

        this.currentField = fieldName;
        if (options.colormap) {
            this.currentColormap = options.colormap;
        }
    }

    /**
     * Set colormap
     * @param {string} colormap - Colormap name
     */
    setColormap(colormap) {
        this.currentColormap = colormap;
        if (this.currentField) {
            this.applyField(this.currentField, { colormap });
        }
    }

    /**
     * Clear field visualization (show solid color)
     */
    clearField() {
        if (this.actors.length > 0) {
            const mapper = this.actors[0].getMapper();
            if (mapper) {
                this.fieldVisualization?.disableScalarVisualization(mapper, true);
            }
        }
        this.currentField = null;
    }

    /**
     * Get available colormap options
     * @returns {Array}
     */
    getColormapOptions() {
        return this.fieldVisualization?.getColormapOptions() || [];
    }

    /**
     * Get available fields
     * @returns {Array}
     */
    getAvailableFields() {
        return this.availableFields;
    }

    /**
     * Setup message listener with field support
     * @private
     * @override
     */
    _setupMessageListener() {
        super._setupMessageListener();

        // Add field-specific message handling
        window.addEventListener('message', (event) => {
            const data = event.data;
            if (!data || !data.type) return;

            switch (data.type) {
                case 'setField':
                    this.applyField(data.field, {
                        colormap: data.colormap,
                        range: data.range
                    });
                    break;

                case 'setColormap':
                    this.setColormap(data.colormap);
                    break;

                case 'clearField':
                    this.clearField();
                    break;
            }
        });
    }
}

/**
 * Create a StandardViewer instance
 * @param {HTMLElement} container
 * @param {Object} config
 * @returns {StandardViewer}
 */
export function createStandardViewer(container, config = {}) {
    const viewer = new StandardViewer(container, config);
    viewer.initialize();
    return viewer;
}

export default StandardViewer;
