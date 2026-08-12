/**
 * TextureManager - Texture handling for GLTF/GLB meshes
 *
 * Manages texture extraction, material fixes, and rendering configuration
 * for properly displaying textured meshes in VTK.js.
 */

/**
 * Texture information structure
 * @typedef {Object} TextureInfo
 * @property {boolean} hasTexture - Whether a texture was found
 * @property {boolean} hasMaterial - Whether PBR material properties exist
 * @property {string} textureType - Type of texture found (baseColor, diffuse, etc.)
 */

export class TextureManager {
    /**
     * Create a TextureManager
     * @param {Object} vtk - VTK.js global object
     * @param {Object} renderWindow - VTK render window
     */
    constructor(vtk, renderWindow) {
        this.vtk = vtk;
        this.renderWindow = renderWindow;
    }

    /**
     * Apply texture fixes to an actor
     * This fixes common issues with GLTF texture rendering in VTK.js
     * @param {Object} actor - VTK actor
     * @returns {TextureInfo} Texture information
     */
    applyTextureFixes(actor) {
        const property = actor.getProperty();
        const mapper = actor.getMapper();
        const info = {
            hasTexture: false,
            hasMaterial: false,
            textureType: null
        };

        if (!property) {
            return info;
        }

        // CRITICAL FIX: Remove metallicRoughnessTexture if present
        // VTK.js GLTFImporter may use this instead of baseColorTexture
        if (typeof property.setRMTexture === 'function') {
            property.setRMTexture(null);
        }
        if (typeof property.setMetallicRoughnessTexture === 'function') {
            property.setMetallicRoughnessTexture(null);
        }

        // Try multiple methods to get the base color texture
        let texture = null;

        if (typeof property.getBaseColorTexture === 'function') {
            texture = property.getBaseColorTexture();
            if (texture) info.textureType = 'baseColor';
        }
        if (!texture && typeof property.getDiffuseTexture === 'function') {
            texture = property.getDiffuseTexture();
            if (texture) info.textureType = 'diffuse';
        }
        if (!texture && typeof property.getTexture === 'function') {
            texture = property.getTexture();
            if (texture) info.textureType = 'generic';
        }

        if (texture) {
            info.hasTexture = true;
            actor.addTexture(texture);
        }

        // Check for PBR material properties
        if (typeof property.getMetallic === 'function' ||
            typeof property.getRoughness === 'function') {
            info.hasMaterial = true;
        }

        // Set material to non-metallic for proper texture display
        if (typeof property.setMetallic === 'function') {
            property.setMetallic(0.0);
        }
        if (typeof property.setRoughness === 'function') {
            property.setRoughness(1.0);
        }
        if (typeof property.setInterpolation === 'function') {
            property.setInterpolation(2); // Phong shading
        }

        // Configure mapper for texture rendering
        if (mapper) {
            mapper.setScalarVisibility(false);
            mapper.modified();
        }

        return info;
    }

    /**
     * Apply texture fixes to multiple actors
     * @param {Object[]} actors - Array of VTK actors
     * @returns {TextureInfo} Combined texture information
     */
    applyTextureFixesMultiple(actors) {
        const combinedInfo = {
            hasTexture: false,
            hasMaterial: false,
            texturedActors: 0
        };

        actors.forEach(actor => {
            const info = this.applyTextureFixes(actor);
            if (info.hasTexture) {
                combinedInfo.hasTexture = true;
                combinedInfo.texturedActors++;
            }
            if (info.hasMaterial) {
                combinedInfo.hasMaterial = true;
            }
        });

        return combinedInfo;
    }

    /**
     * Configure renderer for texture display
     * @param {Object} renderer - VTK renderer
     */
    configureRenderer(renderer) {
        // Enable two-sided lighting for better texture visibility
        if (typeof renderer.setTwoSidedLighting === 'function') {
            renderer.setTwoSidedLighting(true);
        }

        // Disable shadows (can interfere with texture display)
        if (typeof renderer.setUseShadows === 'function') {
            renderer.setUseShadows(false);
        }
    }

    /**
     * Set actor appearance for non-textured display (solid color)
     * @param {Object} actor - VTK actor
     * @param {number[]} color - RGB color array [0-1 range]
     */
    setActorColor(actor, color = [1.0, 1.0, 1.0]) {
        const property = actor.getProperty();
        const mapper = actor.getMapper();

        if (property) {
            property.setColor(...color);
        }

        if (mapper) {
            mapper.setScalarVisibility(false);
        }
    }

    /**
     * Extract texture information from actor without modifying it
     * @param {Object} actor - VTK actor
     * @returns {TextureInfo} Texture information
     */
    getTextureInfo(actor) {
        const property = actor.getProperty();
        const info = {
            hasTexture: false,
            hasMaterial: false,
            textureType: null
        };

        if (!property) {
            return info;
        }

        // Check for textures
        if (typeof property.getBaseColorTexture === 'function' && property.getBaseColorTexture()) {
            info.hasTexture = true;
            info.textureType = 'baseColor';
        } else if (typeof property.getDiffuseTexture === 'function' && property.getDiffuseTexture()) {
            info.hasTexture = true;
            info.textureType = 'diffuse';
        } else if (typeof property.getTexture === 'function' && property.getTexture()) {
            info.hasTexture = true;
            info.textureType = 'generic';
        }

        // Check for material
        if (typeof property.getMetallic === 'function' ||
            typeof property.getRoughness === 'function') {
            info.hasMaterial = true;
        }

        return info;
    }

    /**
     * Check if GLTF texture support is available
     * @returns {boolean}
     */
    isTextureSupported() {
        try {
            return this.vtk.IO.Geometry.vtkGLTFImporter !== undefined;
        } catch {
            return false;
        }
    }
}

export default TextureManager;
