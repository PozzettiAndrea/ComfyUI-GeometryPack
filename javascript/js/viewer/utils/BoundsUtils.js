/**
 * Bounds calculation utilities for VTK viewers
 */

/**
 * Get the center point of bounds
 * @param {number[]} bounds - VTK bounds array [xmin, xmax, ymin, ymax, zmin, zmax]
 * @returns {number[]} Center point [x, y, z]
 */
export function getCenter(bounds) {
    return [
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2
    ];
}

/**
 * Get the maximum dimension of bounds
 * @param {number[]} bounds - VTK bounds array [xmin, xmax, ymin, ymax, zmin, zmax]
 * @returns {number} Maximum dimension
 */
export function getMaxDimension(bounds) {
    return Math.max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    );
}

/**
 * Get all three dimensions of bounds
 * @param {number[]} bounds - VTK bounds array [xmin, xmax, ymin, ymax, zmin, zmax]
 * @returns {number[]} Dimensions [width, height, depth]
 */
export function getDimensions(bounds) {
    return [
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4]
    ];
}

/**
 * Combine multiple bounds into a single bounding box
 * @param {number[][]} boundsArray - Array of VTK bounds
 * @returns {number[]} Combined bounds [xmin, xmax, ymin, ymax, zmin, zmax]
 */
export function combineBounds(boundsArray) {
    if (!boundsArray || boundsArray.length === 0) {
        return [0, 0, 0, 0, 0, 0];
    }

    const combined = [...boundsArray[0]];

    for (let i = 1; i < boundsArray.length; i++) {
        const b = boundsArray[i];
        combined[0] = Math.min(combined[0], b[0]); // xmin
        combined[1] = Math.max(combined[1], b[1]); // xmax
        combined[2] = Math.min(combined[2], b[2]); // ymin
        combined[3] = Math.max(combined[3], b[3]); // ymax
        combined[4] = Math.min(combined[4], b[4]); // zmin
        combined[5] = Math.max(combined[5], b[5]); // zmax
    }

    return combined;
}

/**
 * Check if bounds are valid (non-empty)
 * @param {number[]} bounds - VTK bounds array
 * @returns {boolean} True if bounds are valid
 */
export function isValidBounds(bounds) {
    if (!bounds || bounds.length !== 6) return false;

    // Check for NaN or Infinity
    for (const val of bounds) {
        if (!Number.isFinite(val)) return false;
    }

    // Check that max >= min for all dimensions
    return bounds[1] >= bounds[0] &&
           bounds[3] >= bounds[2] &&
           bounds[5] >= bounds[4];
}

export default {
    getCenter,
    getMaxDimension,
    getDimensions,
    combineBounds,
    isValidBounds
};
