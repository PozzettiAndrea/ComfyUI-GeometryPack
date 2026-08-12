/**
 * LoaderFactory - Auto-detect and load mesh files
 *
 * Provides a unified interface for loading different mesh formats.
 * Automatically detects file type and uses the appropriate loader.
 */

import { loadSTL, canLoadSTL } from './STLLoader.js';
import { loadOBJ, canLoadOBJ } from './OBJLoader.js';
import { loadVTP, canLoadVTP } from './VTPLoader.js';
import { loadGLTF, loadGLTFSimple, canLoadGLTF, isGLTFSupported } from './GLTFLoader.js';

/**
 * Supported format types
 */
export const FormatTypes = {
    STL: 'STL',
    OBJ: 'OBJ',
    VTP: 'VTP',
    GLTF: 'GLTF',
    GLB: 'GLB'
};

/**
 * Detect file format from filepath
 * @param {string} filepath - File path or URL
 * @returns {string|null} Format type or null if unknown
 */
export function detectFormat(filepath) {
    const lower = filepath.toLowerCase();

    if (lower.includes('.stl')) return FormatTypes.STL;
    if (lower.includes('.obj')) return FormatTypes.OBJ;
    if (lower.includes('.vtp')) return FormatTypes.VTP;
    if (lower.includes('.glb')) return FormatTypes.GLB;
    if (lower.includes('.gltf')) return FormatTypes.GLTF;

    return null;
}

/**
 * Check if a format requires special handling (e.g., GLTF needs renderer)
 * @param {string} format - Format type
 * @returns {boolean}
 */
export function requiresRenderer(format) {
    return format === FormatTypes.GLTF || format === FormatTypes.GLB;
}

/**
 * Check if a format supports textures
 * @param {string} format - Format type
 * @returns {boolean}
 */
export function supportsTextures(format) {
    return format === FormatTypes.GLTF || format === FormatTypes.GLB;
}

/**
 * Check if a format supports scalar fields
 * @param {string} format - Format type
 * @returns {boolean}
 */
export function supportsFields(format) {
    return format === FormatTypes.VTP;
}

/**
 * Fetch file data as ArrayBuffer
 * @param {string} filepath - URL to fetch
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchFile(filepath, onProgress = null) {
    const response = await fetch(filepath);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    const total = Number(response.headers.get('Content-Length')) || 0;
    // Fall back to a plain (indeterminate) read if we can't stream or size it.
    if (!response.body || !total || typeof onProgress !== 'function') {
        if (typeof onProgress === 'function') onProgress(null);
        return response.arrayBuffer();
    }
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    onProgress(0, 0, total);
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress(Math.min(1, loaded / total), loaded, total);
    }
    const out = new Uint8Array(loaded);
    let offset = 0;
    for (const c of chunks) { out.set(c, offset); offset += c.length; }
    return out.buffer;
}

/**
 * Load a mesh file with automatic format detection
 * @param {string} filepath - URL to the mesh file
 * @param {Object} vtk - VTK.js global object
 * @param {Object} options - Loading options
 * @param {Object} options.renderer - VTK renderer (required for GLTF with textures)
 * @param {boolean} options.withTextures - Load GLTF with full texture support (default: true)
 * @param {boolean} options.twoSidedLighting - Enable two-sided lighting for GLTF (default: true)
 * @returns {Promise<Object>} Result object with polyData/actors and metadata
 */
export async function loadMesh(filepath, vtk, options = {}) {
    const {
        renderer = null,
        withTextures = true,
        twoSidedLighting = true,
        onProgress = null
    } = options;

    const format = detectFormat(filepath);

    if (!format) {
        throw new Error(`Unsupported file format: ${filepath}`);
    }

    console.log(`[LoaderFactory] Loading ${format} file:`, filepath);

    // GLTF/GLB with texture support requires renderer and URL-based loading
    // (VTK fetches internally, so we can't byte-track it -> indeterminate).
    if ((format === FormatTypes.GLTF || format === FormatTypes.GLB) && withTextures && renderer) {
        if (typeof onProgress === 'function') onProgress(null);
        return loadGLTF(filepath, vtk, renderer, {
            applyTextureFixes: true,
            twoSidedLighting
        });
    }

    // For all other cases, fetch the file first (with download progress)
    const arrayBuffer = await fetchFile(filepath, onProgress);

    switch (format) {
        case FormatTypes.STL:
            return loadSTL(arrayBuffer, vtk);

        case FormatTypes.OBJ:
            return loadOBJ(arrayBuffer, vtk);

        case FormatTypes.VTP:
            return loadVTP(arrayBuffer, vtk);

        case FormatTypes.GLTF:
        case FormatTypes.GLB:
            // Simple GLTF load without texture support
            return loadGLTFSimple(arrayBuffer, vtk);

        default:
            throw new Error(`No loader available for format: ${format}`);
    }
}

/**
 * Load multiple mesh files
 * @param {string[]} filepaths - Array of URLs to mesh files
 * @param {Object} vtk - VTK.js global object
 * @param {Object} options - Loading options (same as loadMesh)
 * @returns {Promise<Object[]>} Array of result objects
 */
export async function loadMeshes(filepaths, vtk, options = {}) {
    return Promise.all(filepaths.map(fp => loadMesh(fp, vtk, options)));
}

/**
 * Create a loader configuration for a specific format
 * Useful for manual loader selection
 * @param {string} format - Format type
 * @returns {Object} Loader configuration
 */
export function getLoaderConfig(format) {
    switch (format) {
        case FormatTypes.STL:
            return {
                load: loadSTL,
                canLoad: canLoadSTL,
                requiresRenderer: false,
                supportsTextures: false,
                supportsFields: false
            };
        case FormatTypes.OBJ:
            return {
                load: loadOBJ,
                canLoad: canLoadOBJ,
                requiresRenderer: false,
                supportsTextures: false,
                supportsFields: false
            };
        case FormatTypes.VTP:
            return {
                load: loadVTP,
                canLoad: canLoadVTP,
                requiresRenderer: false,
                supportsTextures: false,
                supportsFields: true
            };
        case FormatTypes.GLTF:
        case FormatTypes.GLB:
            return {
                load: loadGLTF,
                loadSimple: loadGLTFSimple,
                canLoad: canLoadGLTF,
                isSupported: isGLTFSupported,
                requiresRenderer: true,
                supportsTextures: true,
                supportsFields: false
            };
        default:
            return null;
    }
}

export default {
    loadMesh,
    loadMeshes,
    detectFormat,
    requiresRenderer,
    supportsTextures,
    supportsFields,
    getLoaderConfig,
    FormatTypes
};
