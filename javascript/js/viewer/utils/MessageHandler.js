/**
 * PostMessage communication utilities for VTK viewers
 */

/**
 * Message types used by the viewer system
 */
export const MessageTypes = {
    // Incoming messages (from parent)
    LOAD_MESH: 'LOAD_MESH',
    LOAD_DUAL_MESH: 'LOAD_DUAL_MESH',

    // Outgoing messages (to parent)
    MESH_LOADED: 'MESH_LOADED',
    MESH_ERROR: 'MESH_ERROR',
    SCREENSHOT: 'SCREENSHOT'
};

/**
 * Send a message to the parent window
 * @param {string} type - Message type
 * @param {Object} data - Additional data to send
 */
export function sendToParent(type, data = {}) {
    if (window.parent && window.parent !== window) {
        window.parent.postMessage({
            type,
            ...data
        }, '*');
    }
}

/**
 * Send an error message to the parent window
 * @param {string|Error} error - Error message or Error object
 */
export function sendError(error) {
    const message = error instanceof Error ? error.message : error;
    sendToParent(MessageTypes.MESH_ERROR, { error: message });
}

/**
 * Send a screenshot to the parent window
 * @param {string} imageDataUrl - Base64 PNG data URL
 */
export function sendScreenshot(imageDataUrl) {
    sendToParent(MessageTypes.SCREENSHOT, {
        image: imageDataUrl,
        timestamp: Date.now()
    });
}

/**
 * Send mesh loaded confirmation to parent
 */
export function sendMeshLoaded() {
    sendToParent(MessageTypes.MESH_LOADED, { error: null });
}

/**
 * Create a message listener for the viewer
 * @param {Object} handlers - Object mapping message types to handler functions
 * @returns {Function} Cleanup function to remove the listener
 */
export function createMessageListener(handlers) {
    const listener = (event) => {
        const { type } = event.data || {};

        if (type && handlers[type]) {
            handlers[type](event.data);
        }
    };

    window.addEventListener('message', listener);

    // Return cleanup function
    return () => {
        window.removeEventListener('message', listener);
    };
}

/**
 * Extract filename from a filepath URL
 * @param {string} filepath - URL or path to extract filename from
 * @returns {string} Extracted filename
 */
export function extractFilename(filepath) {
    const urlParams = new URLSearchParams(filepath.split('?')[1] || '');
    return urlParams.get('filename') || filepath.split('/').pop().split('?')[0];
}

export default {
    MessageTypes,
    sendToParent,
    sendError,
    sendScreenshot,
    sendMeshLoaded,
    createMessageListener,
    extractFilename
};
