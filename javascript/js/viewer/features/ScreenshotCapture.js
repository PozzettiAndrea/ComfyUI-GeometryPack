/**
 * ScreenshotCapture - Screenshot functionality for VTK viewers
 *
 * Handles capturing screenshots from VTK render windows and
 * sending them via postMessage or triggering downloads.
 */

import { sendScreenshot } from '../utils/MessageHandler.js';

export class ScreenshotCapture {
    /**
     * Create a ScreenshotCapture instance
     * @param {Object} renderWindow - VTK render window
     */
    constructor(renderWindow) {
        this.renderWindow = renderWindow;
    }

    /**
     * Capture a screenshot as a base64 PNG data URL
     * @returns {Promise<string>} Base64 PNG data URL
     */
    async capture() {
        const images = await this.renderWindow.captureImages();
        return images[0]; // Returns base64 PNG data URL
    }

    /**
     * Capture and send screenshot to parent window via postMessage
     * @returns {Promise<void>}
     */
    async captureAndSend() {
        try {
            const image = await this.capture();
            sendScreenshot(image);
            console.log('[ScreenshotCapture] Screenshot captured and sent to parent');
        } catch (error) {
            console.error('[ScreenshotCapture] Error capturing screenshot:', error);
            throw error;
        }
    }

    /**
     * Capture and download screenshot as a file
     * @param {string} filename - Download filename (default: "screenshot.png")
     * @returns {Promise<void>}
     */
    async captureAndDownload(filename = 'screenshot.png') {
        try {
            const image = await this.capture();

            // Create download link
            const link = document.createElement('a');
            link.href = image;
            link.download = filename;
            link.style.display = 'none';

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            console.log('[ScreenshotCapture] Screenshot downloaded:', filename);
        } catch (error) {
            console.error('[ScreenshotCapture] Error downloading screenshot:', error);
            throw error;
        }
    }

    /**
     * Capture screenshot and return as Blob
     * @returns {Promise<Blob>} PNG Blob
     */
    async captureAsBlob() {
        const image = await this.capture();

        // Convert base64 to blob
        const response = await fetch(image);
        return response.blob();
    }

    /**
     * Generate a timestamp-based filename
     * @param {string} prefix - Filename prefix (default: "screenshot")
     * @param {string} extension - File extension (default: "png")
     * @returns {string} Filename with timestamp
     */
    static generateFilename(prefix = 'screenshot', extension = 'png') {
        const now = new Date();
        const timestamp = now.toISOString()
            .replace(/[:.]/g, '-')
            .replace('T', '_')
            .slice(0, 19);
        return `${prefix}_${timestamp}.${extension}`;
    }
}

export default ScreenshotCapture;
