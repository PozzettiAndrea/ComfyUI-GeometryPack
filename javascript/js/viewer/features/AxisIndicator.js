/**
 * AxisIndicator - Orientation widget for VTK viewers
 *
 * Creates and manages an annotated cube axis indicator that shows
 * the current camera orientation.
 */

import { rgbToString } from '../utils/ColorUtils.js';

/**
 * Default axis colors
 */
const DEFAULT_COLORS = {
    x: '#ff5555',  // Red for X axis
    y: '#55ff55',  // Green for Y axis
    z: '#5555ff'   // Blue for Z axis
};

/**
 * Default widget configuration
 */
const DEFAULT_CONFIG = {
    enabled: false,
    corner: 'BOTTOM_LEFT',  // BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT
    viewportSize: 0.15,
    minPixelSize: 100,
    maxPixelSize: 300,
    faceColor: rgbToString(0.5, 0.5, 0.5)
};

export class AxisIndicator {
    /**
     * Create an AxisIndicator
     * @param {Object} vtk - VTK.js global object
     * @param {Object} interactor - VTK interactor (from renderWindow.getInteractor())
     * @param {Object} config - Configuration options
     */
    constructor(vtk, interactor, config = {}) {
        this.vtk = vtk;
        this.interactor = interactor;
        this.config = { ...DEFAULT_CONFIG, ...config };

        // VTK classes
        this.vtkOrientationMarkerWidget = vtk.Interaction.Widgets.vtkOrientationMarkerWidget;
        this.vtkAnnotatedCubeActor = vtk.Rendering.Core.vtkAnnotatedCubeActor;

        // Widget state
        this.widget = null;
        this.cubeActor = null;
        this.enabled = this.config.enabled;

        this._createWidget();
    }

    /**
     * Create the orientation widget
     * @private
     */
    _createWidget() {
        // Create annotated cube actor
        this.cubeActor = this.vtkAnnotatedCubeActor.newInstance();

        // Set default style
        this.cubeActor.setDefaultStyle({
            text: '+X',
            fontStyle: 'bold',
            fontFamily: 'Arial',
            fontColor: 'white',
            fontSizeScale: (res) => res / 2,
            faceColor: this.config.faceColor,
            faceRotation: 0,
            edgeThickness: 0.1,
            edgeColor: 'white',
            resolution: 400,
        });

        // Customize each face with axis colors
        this.cubeActor.setXPlusFaceProperty({ text: '+X', faceColor: DEFAULT_COLORS.x });
        this.cubeActor.setXMinusFaceProperty({ text: '-X', faceColor: DEFAULT_COLORS.x });
        this.cubeActor.setYPlusFaceProperty({ text: '+Y', faceColor: DEFAULT_COLORS.y });
        this.cubeActor.setYMinusFaceProperty({ text: '-Y', faceColor: DEFAULT_COLORS.y });
        this.cubeActor.setZPlusFaceProperty({ text: '+Z', faceColor: DEFAULT_COLORS.z });
        this.cubeActor.setZMinusFaceProperty({ text: '-Z', faceColor: DEFAULT_COLORS.z });

        // Create orientation marker widget
        this.widget = this.vtkOrientationMarkerWidget.newInstance({
            actor: this.cubeActor,
            interactor: this.interactor,
        });

        // Configure widget position and size
        this._applyCornerConfig();
        this.widget.setViewportSize(this.config.viewportSize);
        this.widget.setMinPixelSize(this.config.minPixelSize);
        this.widget.setMaxPixelSize(this.config.maxPixelSize);

        // Set initial enabled state
        this.widget.setEnabled(this.enabled);
    }

    /**
     * Apply corner configuration
     * @private
     */
    _applyCornerConfig() {
        const Corners = this.vtkOrientationMarkerWidget.Corners;
        const cornerMap = {
            'BOTTOM_LEFT': Corners.BOTTOM_LEFT,
            'BOTTOM_RIGHT': Corners.BOTTOM_RIGHT,
            'TOP_LEFT': Corners.TOP_LEFT,
            'TOP_RIGHT': Corners.TOP_RIGHT
        };

        const corner = cornerMap[this.config.corner] || Corners.BOTTOM_LEFT;
        this.widget.setViewportCorner(corner);
    }

    /**
     * Enable the axis indicator
     */
    enable() {
        this.enabled = true;
        if (this.widget) {
            this.widget.setEnabled(true);
        }
    }

    /**
     * Disable the axis indicator
     */
    disable() {
        this.enabled = false;
        if (this.widget) {
            this.widget.setEnabled(false);
        }
    }

    /**
     * Toggle the axis indicator
     * @returns {boolean} New enabled state
     */
    toggle() {
        if (this.enabled) {
            this.disable();
        } else {
            this.enable();
        }
        return this.enabled;
    }

    /**
     * Set enabled state
     * @param {boolean} enabled - Whether to enable the indicator
     */
    setEnabled(enabled) {
        if (enabled) {
            this.enable();
        } else {
            this.disable();
        }
    }

    /**
     * Check if enabled
     * @returns {boolean}
     */
    isEnabled() {
        return this.enabled;
    }

    /**
     * Set the corner position
     * @param {string} corner - 'BOTTOM_LEFT', 'BOTTOM_RIGHT', 'TOP_LEFT', 'TOP_RIGHT'
     */
    setCorner(corner) {
        this.config.corner = corner;
        this._applyCornerConfig();
    }

    /**
     * Set viewport size (0-1 range)
     * @param {number} size - Viewport size as fraction
     */
    setViewportSize(size) {
        this.config.viewportSize = size;
        if (this.widget) {
            this.widget.setViewportSize(size);
        }
    }

    /**
     * Set axis colors
     * @param {Object} colors - Color object with x, y, z properties (hex strings)
     */
    setAxisColors(colors) {
        if (colors.x) {
            this.cubeActor.setXPlusFaceProperty({ faceColor: colors.x });
            this.cubeActor.setXMinusFaceProperty({ faceColor: colors.x });
        }
        if (colors.y) {
            this.cubeActor.setYPlusFaceProperty({ faceColor: colors.y });
            this.cubeActor.setYMinusFaceProperty({ faceColor: colors.y });
        }
        if (colors.z) {
            this.cubeActor.setZPlusFaceProperty({ faceColor: colors.z });
            this.cubeActor.setZMinusFaceProperty({ faceColor: colors.z });
        }
    }

    /**
     * Get the underlying VTK widget
     * @returns {Object} VTK orientation marker widget
     */
    getWidget() {
        return this.widget;
    }

    /**
     * Get the cube actor
     * @returns {Object} VTK annotated cube actor
     */
    getCubeActor() {
        return this.cubeActor;
    }

    /**
     * Cleanup resources
     */
    destroy() {
        if (this.widget) {
            this.widget.setEnabled(false);
            this.widget.delete();
            this.widget = null;
        }
        if (this.cubeActor) {
            this.cubeActor.delete();
            this.cubeActor = null;
        }
    }
}

export default AxisIndicator;
