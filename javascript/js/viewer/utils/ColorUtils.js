/**
 * Color utility functions for VTK viewers
 */

/**
 * Convert RGB values (0-1 range) to CSS rgb() string
 * @param {number} r - Red component (0-1)
 * @param {number} g - Green component (0-1)
 * @param {number} b - Blue component (0-1)
 * @returns {string} CSS rgb string like "rgb(255, 128, 0)"
 */
export function rgbToString(r, g, b) {
    const r255 = Math.round(r * 255);
    const g255 = Math.round(g * 255);
    const b255 = Math.round(b * 255);
    return `rgb(${r255}, ${g255}, ${b255})`;
}

/**
 * Convert hex color string to RGB array (0-1 range)
 * @param {string} hex - Hex color like "#ffffff" or "#fff"
 * @returns {number[]} RGB array [r, g, b] with values 0-1
 */
export function hexToRgb(hex) {
    // Remove # if present
    hex = hex.replace(/^#/, '');

    // Handle shorthand hex (#fff -> #ffffff)
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }

    const r = parseInt(hex.substr(0, 2), 16) / 255;
    const g = parseInt(hex.substr(2, 2), 16) / 255;
    const b = parseInt(hex.substr(4, 2), 16) / 255;

    return [r, g, b];
}

/**
 * Convert RGB array (0-1 range) to hex color string
 * @param {number} r - Red component (0-1)
 * @param {number} g - Green component (0-1)
 * @param {number} b - Blue component (0-1)
 * @returns {string} Hex color like "#ffffff"
 */
export function rgbToHex(r, g, b) {
    const toHex = (c) => {
        const hex = Math.round(c * 255).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    };
    return '#' + toHex(r) + toHex(g) + toHex(b);
}

export default {
    rgbToString,
    hexToRgb,
    rgbToHex
};
