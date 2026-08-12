/**
 * CameraController - Unified camera management for VTK viewers
 *
 * Handles camera positioning, view presets, and projection modes.
 * Configurable distance multiplier allows different viewers to use different defaults.
 */

import { getCenter, getMaxDimension } from '../utils/BoundsUtils.js';

/**
 * View direction configurations
 * Maps direction strings to camera position offsets and viewUp vectors
 */
const VIEW_CONFIGS = {
    '+X': { offset: [1, 0, 0], viewUp: [0, 0, 1] },
    '-X': { offset: [-1, 0, 0], viewUp: [0, 0, 1] },
    '+Y': { offset: [0, 1, 0], viewUp: [0, 0, 1] },
    '-Y': { offset: [0, -1, 0], viewUp: [0, 0, 1] },
    '+Z': { offset: [0, 0, 1], viewUp: [0, 1, 0] },
    '-Z': { offset: [0, 0, -1], viewUp: [0, 1, 0] }
};

/**
 * Default camera configuration
 */
const DEFAULT_CONFIG = {
    distanceMultiplier: 2.5,  // Standard viewers use 2.5, textured uses 2.0
    initialViewOffset: [1, 1, 1],  // Diagonal view for initial positioning
    initialViewUp: [0, 0, 1]
};

export class CameraController {
    /**
     * Create a CameraController
     * @param {Object} renderer - VTK renderer instance
     * @param {Object} renderWindow - VTK render window instance
     * @param {Object} config - Configuration options
     * @param {number} config.distanceMultiplier - Camera distance = maxDim * multiplier (default: 2.5)
     */
    constructor(renderer, renderWindow, config = {}) {
        this.renderer = renderer;
        this.renderWindow = renderWindow;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.currentBounds = null;
    }

    /**
     * Get the active camera from the renderer
     * @returns {Object} VTK camera instance
     */
    getCamera() {
        return this.renderer.getActiveCamera();
    }

    /**
     * Store bounds for later use (e.g., view presets)
     * @param {number[]} bounds - VTK bounds [xmin, xmax, ymin, ymax, zmin, zmax]
     */
    setBounds(bounds) {
        this.currentBounds = bounds;
    }

    /**
     * Position camera for initial viewing of mesh
     * @param {number[]} bounds - VTK bounds [xmin, xmax, ymin, ymax, zmin, zmax]
     */
    positionInitialCamera(bounds) {
        this.currentBounds = bounds;

        const camera = this.getCamera();
        const center = getCenter(bounds);
        const maxDim = getMaxDimension(bounds);
        const distance = maxDim * this.config.distanceMultiplier;

        const offset = this.config.initialViewOffset;
        const position = [
            center[0] + offset[0] * distance,
            center[1] + offset[1] * distance,
            center[2] + offset[2] * distance
        ];

        camera.setPosition(...position);
        camera.setFocalPoint(...center);
        camera.setViewUp(...this.config.initialViewUp);

        this.renderer.resetCamera();
        this.renderWindow.render();
    }

    /**
     * Set camera to a specific axis-aligned view
     * @param {string} direction - One of '+X', '-X', '+Y', '-Y', '+Z', '-Z'
     * @param {number[]} bounds - Optional bounds, uses stored bounds if not provided
     */
    setCameraView(direction, bounds = null) {
        const activeBounds = bounds || this.currentBounds;
        if (!activeBounds) {
            console.warn('[CameraController] No bounds available for setCameraView');
            return;
        }

        const viewConfig = VIEW_CONFIGS[direction];
        if (!viewConfig) {
            console.warn(`[CameraController] Unknown view direction: ${direction}`);
            return;
        }

        const camera = this.getCamera();
        const center = getCenter(activeBounds);
        const maxDim = getMaxDimension(activeBounds);
        const distance = maxDim * this.config.distanceMultiplier;

        const position = [
            center[0] + viewConfig.offset[0] * distance,
            center[1] + viewConfig.offset[1] * distance,
            center[2] + viewConfig.offset[2] * distance
        ];

        camera.setPosition(...position);
        camera.setFocalPoint(...center);
        camera.setViewUp(...viewConfig.viewUp);

        this.renderer.resetCamera();
        this.renderWindow.render();
    }

    /**
     * Reset camera to fit the current bounds
     */
    resetCamera() {
        if (this.currentBounds) {
            this.renderer.resetCamera(this.currentBounds);
        } else {
            this.renderer.resetCamera();
        }
        this.renderWindow.render();
    }

    /**
     * Reset camera clipping range (useful after mesh updates)
     */
    resetClippingRange() {
        this.renderer.resetCameraClippingRange();
        this.renderWindow.render();
    }

    /**
     * Set parallel (orthographic) vs perspective projection
     * @param {boolean} enabled - True for parallel, false for perspective
     */
    setParallelProjection(enabled) {
        const camera = this.getCamera();
        camera.setParallelProjection(enabled);
        this.renderWindow.render();
    }

    /**
     * Get current parallel projection state
     * @returns {boolean} True if parallel projection is enabled
     */
    getParallelProjection() {
        return this.getCamera().getParallelProjection();
    }

    /**
     * Synchronize this camera with another camera
     * Useful for side-by-side viewers
     * @param {Object} sourceCamera - Camera to copy state from
     */
    synchronizeFrom(sourceCamera) {
        const camera = this.getCamera();
        camera.setPosition(...sourceCamera.getPosition());
        camera.setFocalPoint(...sourceCamera.getFocalPoint());
        camera.setViewUp(...sourceCamera.getViewUp());
        camera.setParallelProjection(sourceCamera.getParallelProjection());
        camera.setClippingRange(...sourceCamera.getClippingRange());
        camera.setParallelScale(sourceCamera.getParallelScale());
        camera.setViewAngle(sourceCamera.getViewAngle());
    }

    /**
     * Get camera state for serialization
     * @returns {Object} Camera state object
     */
    getState() {
        const camera = this.getCamera();
        return {
            position: camera.getPosition(),
            focalPoint: camera.getFocalPoint(),
            viewUp: camera.getViewUp(),
            parallelProjection: camera.getParallelProjection(),
            clippingRange: camera.getClippingRange(),
            parallelScale: camera.getParallelScale(),
            viewAngle: camera.getViewAngle()
        };
    }

    /**
     * Restore camera state from serialized object
     * @param {Object} state - Camera state object from getState()
     */
    setState(state) {
        const camera = this.getCamera();
        if (state.position) camera.setPosition(...state.position);
        if (state.focalPoint) camera.setFocalPoint(...state.focalPoint);
        if (state.viewUp) camera.setViewUp(...state.viewUp);
        if (state.parallelProjection !== undefined) camera.setParallelProjection(state.parallelProjection);
        if (state.clippingRange) camera.setClippingRange(...state.clippingRange);
        if (state.parallelScale !== undefined) camera.setParallelScale(state.parallelScale);
        if (state.viewAngle !== undefined) camera.setViewAngle(state.viewAngle);
        this.renderWindow.render();
    }

    /**
     * Focus camera on a specific 3D point
     * Moves the camera to look at the point while maintaining relative view direction
     * @param {number[]} point - [x, y, z] coordinates to focus on
     * @param {number} viewRadius - Optional radius around point to view (defaults to 1/10 of mesh size)
     */
    focusOnPoint(point, viewRadius = null) {
        const camera = this.getCamera();

        // Get current camera direction (from focal point to camera position)
        const currentFocalPoint = camera.getFocalPoint();
        const currentPosition = camera.getPosition();
        const direction = [
            currentPosition[0] - currentFocalPoint[0],
            currentPosition[1] - currentFocalPoint[1],
            currentPosition[2] - currentFocalPoint[2]
        ];

        // Normalize direction
        const length = Math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2);
        if (length > 0) {
            direction[0] /= length;
            direction[1] /= length;
            direction[2] /= length;
        } else {
            // Default to looking from +Z if no current direction
            direction[0] = 0;
            direction[1] = 0;
            direction[2] = 1;
        }

        // Calculate view distance
        let distance;
        if (viewRadius !== null) {
            distance = viewRadius * this.config.distanceMultiplier;
        } else if (this.currentBounds) {
            const maxDim = getMaxDimension(this.currentBounds);
            distance = maxDim * 0.3;  // Closer view for focused point
        } else {
            distance = 10;  // Fallback
        }

        // Position camera along the direction from the point
        const newPosition = [
            point[0] + direction[0] * distance,
            point[1] + direction[1] * distance,
            point[2] + direction[2] * distance
        ];

        camera.setFocalPoint(...point);
        camera.setPosition(...newPosition);

        this.renderer.resetCameraClippingRange();
        this.renderWindow.render();
    }
}

export default CameraController;
