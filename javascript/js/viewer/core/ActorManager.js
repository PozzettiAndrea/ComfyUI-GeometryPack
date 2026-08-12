/**
 * ActorManager - Unified actor/property management for VTK viewers
 *
 * Handles actor configuration, appearance settings, edge visibility,
 * and material properties. Works with both single actors and actor arrays.
 */

import { hexToRgb } from '../utils/ColorUtils.js';

/**
 * Default actor appearance settings
 */
const DEFAULT_SETTINGS = {
    color: [1.0, 1.0, 1.0],     // White
    edgeColor: [0.0, 0.0, 0.0], // Black edges
    ambient: 0.4,
    diffuse: 0.8,
    specular: 0.3,
    specularPower: 20,
    edgeVisibility: false,
    lineWidth: 1,
    pointSize: 5,
    representation: 2,  // 0=Points, 1=Wireframe, 2=Surface
    opacity: 1.0
};

/**
 * Representation mode constants
 */
export const Representation = {
    POINTS: 0,
    WIREFRAME: 1,
    SURFACE: 2
};

export class ActorManager {
    /**
     * Create an ActorManager
     * @param {Object} renderWindow - VTK render window for triggering renders
     */
    constructor(renderWindow) {
        this.renderWindow = renderWindow;
        this.actors = [];
        this.settings = { ...DEFAULT_SETTINGS };
    }

    /**
     * Add an actor to be managed
     * @param {Object} actor - VTK actor instance
     */
    addActor(actor) {
        this.actors.push(actor);
        this.configureActor(actor, this.settings);
    }

    /**
     * Remove an actor from management
     * @param {Object} actor - VTK actor instance
     */
    removeActor(actor) {
        const index = this.actors.indexOf(actor);
        if (index > -1) {
            this.actors.splice(index, 1);
        }
    }

    /**
     * Clear all managed actors
     */
    clearActors() {
        this.actors = [];
    }

    /**
     * Set actors to manage (replaces existing)
     * @param {Object[]} actors - Array of VTK actor instances
     */
    setActors(actors) {
        this.actors = Array.isArray(actors) ? actors : [actors];
        this.actors.forEach(actor => this.configureActor(actor, this.settings));
    }

    /**
     * Configure a single actor with settings
     * @param {Object} actor - VTK actor instance
     * @param {Object} settings - Settings object
     */
    configureActor(actor, settings = {}) {
        const merged = { ...this.settings, ...settings };
        const property = actor.getProperty();

        if (!property) {
            console.warn('[ActorManager] Actor has no property object');
            return;
        }

        // Color (supports hex string or RGB array)
        if (merged.color) {
            const rgb = typeof merged.color === 'string'
                ? hexToRgb(merged.color)
                : merged.color;
            property.setColor(...rgb);
        }

        // Edge color
        if (merged.edgeColor) {
            const rgb = typeof merged.edgeColor === 'string'
                ? hexToRgb(merged.edgeColor)
                : merged.edgeColor;
            property.setEdgeColor(...rgb);
        }

        // Lighting properties
        if (merged.ambient !== undefined) property.setAmbient(merged.ambient);
        if (merged.diffuse !== undefined) property.setDiffuse(merged.diffuse);
        if (merged.specular !== undefined) property.setSpecular(merged.specular);
        if (merged.specularPower !== undefined) property.setSpecularPower(merged.specularPower);

        // Edge and line properties
        if (merged.edgeVisibility !== undefined) property.setEdgeVisibility(merged.edgeVisibility);
        if (merged.lineWidth !== undefined) property.setLineWidth(merged.lineWidth);

        // Point properties
        if (merged.pointSize !== undefined) property.setPointSize(merged.pointSize);

        // Representation mode
        if (merged.representation !== undefined) property.setRepresentation(merged.representation);

        // Opacity
        if (merged.opacity !== undefined) property.setOpacity(merged.opacity);
    }

    /**
     * Apply settings to all managed actors
     * @param {Object} settings - Settings object
     * @param {boolean} render - Whether to trigger a render (default: true)
     */
    applySettings(settings, render = true) {
        this.settings = { ...this.settings, ...settings };
        this.actors.forEach(actor => this.configureActor(actor, this.settings));
        if (render) {
            this.renderWindow.render();
        }
    }

    /**
     * Set edge visibility on all managed actors
     * @param {boolean} visible - Whether edges should be visible
     */
    setEdgeVisibility(visible) {
        this.applySettings({ edgeVisibility: visible });
    }

    /**
     * Set color on all managed actors
     * @param {string|number[]} color - Hex string or RGB array (0-1 range)
     */
    setColor(color) {
        this.applySettings({ color });
    }

    /**
     * Set opacity on all managed actors
     * @param {number} opacity - Opacity value (0-1)
     */
    setOpacity(opacity) {
        this.applySettings({ opacity });
    }

    /**
     * Set representation mode on all managed actors
     * @param {number} mode - 0=Points, 1=Wireframe, 2=Surface
     */
    setRepresentation(mode) {
        this.applySettings({ representation: mode });
    }

    /**
     * Set line width for edges/wireframe
     * @param {number} width - Line width in pixels
     */
    setLineWidth(width) {
        this.applySettings({ lineWidth: width });
    }

    /**
     * Set point size for point cloud rendering
     * @param {number} size - Point size in pixels
     */
    setPointSize(size) {
        this.applySettings({ pointSize: size });
    }

    /**
     * Get current settings
     * @returns {Object} Current settings object
     */
    getSettings() {
        return { ...this.settings };
    }

    /**
     * Apply settings from a settings panel (handles hex colors and number parsing)
     * @param {Object} panelValues - Values from settings panel inputs
     */
    applyPanelSettings(panelValues) {
        const settings = {};

        if (panelValues.meshColor !== undefined) {
            settings.color = panelValues.meshColor;
        }
        if (panelValues.pointSize !== undefined) {
            settings.pointSize = parseFloat(panelValues.pointSize);
        }
        if (panelValues.edgeWidth !== undefined) {
            settings.lineWidth = parseFloat(panelValues.edgeWidth);
        }
        if (panelValues.representation !== undefined) {
            settings.representation = parseInt(panelValues.representation, 10);
        }
        if (panelValues.opacity !== undefined) {
            settings.opacity = parseFloat(panelValues.opacity);
        }

        this.applySettings(settings);
    }

    /**
     * Configure for textured mode (disables scalar visibility, sets material)
     * @param {Object} actor - VTK actor instance
     */
    configureForTexture(actor) {
        const property = actor.getProperty();
        const mapper = actor.getMapper();

        if (property) {
            // Material settings for proper texture display
            if (typeof property.setMetallic === 'function') {
                property.setMetallic(0.0);
            }
            if (typeof property.setRoughness === 'function') {
                property.setRoughness(1.0);
            }
            if (typeof property.setInterpolation === 'function') {
                property.setInterpolation(2); // Phong shading
            }

            // Remove metallic/roughness texture if present (common GLTF issue)
            if (typeof property.setRMTexture === 'function') {
                property.setRMTexture(null);
            }
            if (typeof property.setMetallicRoughnessTexture === 'function') {
                property.setMetallicRoughnessTexture(null);
            }
        }

        if (mapper) {
            mapper.setScalarVisibility(false);
            mapper.modified();
        }
    }

    /**
     * Enable/disable scalar (field) visualization on mapper
     * @param {Object} mapper - VTK mapper instance
     * @param {boolean} enabled - Whether to show scalar colors
     */
    setScalarVisibility(mapper, enabled) {
        if (mapper) {
            mapper.setScalarVisibility(enabled);
            mapper.modified();
        }
    }
}

export default ActorManager;
