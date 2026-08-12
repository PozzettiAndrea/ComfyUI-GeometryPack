/**
 * FieldVisualization - Scalar field coloring for VTK viewers
 *
 * Handles scalar field visualization including:
 * - Field detection (point/cell data)
 * - Colormap application
 * - Range calculation (auto/manual, single/synchronized)
 */

/**
 * Popular colormaps to show at top of selector
 */
const POPULAR_COLORMAPS = [
    'erdc_rainbow_bright',
    'Cool to Warm',
    'Viridis (matplotlib)',
    'Plasma (matplotlib)',
    'Inferno (matplotlib)',
    'Magma (matplotlib)',
    'jet',
    'rainbow'
];

export class FieldVisualization {
    /**
     * Create a FieldVisualization instance
     * @param {Object} vtk - VTK.js global object
     * @param {Object} renderWindow - VTK render window for triggering renders
     */
    constructor(vtk, renderWindow) {
        this.vtk = vtk;
        this.renderWindow = renderWindow;

        // VTK classes
        this.vtkColorTransferFunction = vtk.Rendering.Core.vtkColorTransferFunction;
        this.vtkColorMaps = vtk.Rendering.Core.vtkColorTransferFunction.vtkColorMaps;

        // State
        this.currentField = null;
        this.currentColormap = 'erdc_rainbow_bright';
        this.autoRange = true;
        this.manualRange = [0, 1];
    }

    /**
     * Extract field information from polydata
     * @param {Object} polyData - VTK polydata object
     * @returns {Object} Field information with pointFields and cellFields arrays
     */
    extractFields(polyData) {
        if (!polyData) {
            return { pointFields: [], cellFields: [], allFields: [] };
        }

        // Fields to hide from dropdown (VTK internal arrays that aren't useful to visualize)
        const HIDDEN_FIELDS = ['Attribute', 'vtkOriginalPointIds', 'vtkOriginalCellIds'];

        const pointData = polyData.getPointData();
        const cellData = polyData.getCellData();

        const pointFields = [];
        const cellFields = [];

        // Extract point (vertex) data arrays
        for (let i = 0; i < pointData.getNumberOfArrays(); i++) {
            const array = pointData.getArray(i);
            const name = array.getName();
            // Skip hidden/internal fields
            if (HIDDEN_FIELDS.includes(name)) continue;
            const range = array.getRange();
            pointFields.push({
                name,
                displayName: `${name} (vertex)`,
                type: 'point',
                fullName: `point:${name}`,
                numComponents: array.getNumberOfComponents(),
                range: [range[0], range[1]]
            });
        }

        // Extract cell (face) data arrays
        for (let i = 0; i < cellData.getNumberOfArrays(); i++) {
            const array = cellData.getArray(i);
            const name = array.getName();
            // Skip hidden/internal fields
            if (HIDDEN_FIELDS.includes(name)) continue;
            const range = array.getRange();
            cellFields.push({
                name,
                displayName: `${name} (face)`,
                type: 'cell',
                fullName: `cell:${name}`,
                numComponents: array.getNumberOfComponents(),
                range: [range[0], range[1]]
            });
        }

        const allFields = [...pointFields, ...cellFields];

        return {
            pointFields,
            cellFields,
            allFields,
            hasFields: allFields.length > 0
        };
    }

    /**
     * Apply a scalar field to a mapper
     * @param {Object} mapper - VTK mapper
     * @param {Object} polyData - VTK polydata
     * @param {string} fieldName - Field name (e.g., "point:temperature" or "cell:pressure")
     * @param {Object} options - Options
     * @param {number[]} options.range - Manual range [min, max] (overrides auto)
     * @param {string} options.colormap - Colormap name
     * @param {boolean} options.render - Whether to trigger render (default: true)
     */
    applyField(mapper, polyData, fieldName, options = {}) {
        const {
            range = null,
            colormap = this.currentColormap,
            render = true
        } = options;

        if (!fieldName || fieldName === 'none' || !polyData) {
            this.disableScalarVisualization(mapper, render);
            return;
        }

        // Parse field name
        const isCellData = fieldName.startsWith('cell:');
        const actualFieldName = isCellData
            ? fieldName.substring(5)
            : fieldName.startsWith('point:')
                ? fieldName.substring(6)
                : fieldName;

        // Get data array
        const data = isCellData ? polyData.getCellData() : polyData.getPointData();
        const scalarArray = data.getArrayByName(actualFieldName);

        if (!scalarArray) {
            console.warn(`[FieldVisualization] Field "${actualFieldName}" not found`);
            this.disableScalarVisualization(mapper, render);
            return;
        }

        // Get range
        const dataRange = scalarArray.getRange();
        const [minVal, maxVal] = range || (this.autoRange ? dataRange : this.manualRange);

        // Check for degenerate range
        const epsilon = 1e-10;
        if (Math.abs(maxVal - minVal) < epsilon && !range) {
            console.warn(`[FieldVisualization] Field "${actualFieldName}" has zero-width range`);
            this.disableScalarVisualization(mapper, render);
            return;
        }

        // Set active scalars
        data.setActiveScalars(actualFieldName);

        // Configure mapper
        mapper.setScalarVisibility(true);
        if (isCellData) {
            mapper.setScalarModeToUseCellData();
        } else {
            mapper.setScalarModeToUsePointData();
        }
        mapper.setScalarRange(minVal, maxVal);

        // Apply colormap
        this.applyColormap(mapper, colormap, minVal, maxVal);

        // Update state
        this.currentField = fieldName;
        this.currentColormap = colormap;

        // Update mapper to apply changes
        mapper.update();

        if (render) {
            this.renderWindow.render();
        }

        console.log(`[FieldVisualization] Applied field "${actualFieldName}" with range [${minVal.toFixed(3)}, ${maxVal.toFixed(3)}]`);
    }

    /**
     * Apply a colormap to a mapper
     * @param {Object} mapper - VTK mapper
     * @param {string} colormapName - Colormap preset name
     * @param {number} minVal - Range minimum
     * @param {number} maxVal - Range maximum
     */
    applyColormap(mapper, colormapName, minVal, maxVal) {
        try {
            const lookupTable = this.vtkColorTransferFunction.newInstance();
            const preset = this.vtkColorMaps.getPresetByName(colormapName);

            if (preset) {
                lookupTable.applyColorMap(preset);
                lookupTable.setMappingRange(minVal, maxVal);
                lookupTable.updateRange();
                mapper.setLookupTable(lookupTable);
            } else {
                console.warn(`[FieldVisualization] Colormap "${colormapName}" not found`);
            }
        } catch (error) {
            console.error('[FieldVisualization] Error applying colormap:', error);
        }
    }

    /**
     * Disable scalar visualization on a mapper
     * @param {Object} mapper - VTK mapper
     * @param {boolean} render - Whether to trigger render
     */
    disableScalarVisualization(mapper, render = true) {
        if (mapper) {
            mapper.setScalarVisibility(false);
        }
        this.currentField = null;
        if (render) {
            this.renderWindow.render();
        }
    }

    /**
     * Get available colormap presets
     * @param {boolean} popularFirst - Put popular colormaps at top (default: true)
     * @returns {Array} Array of {value, label, isPopular} objects
     */
    getColormapOptions(popularFirst = true) {
        const allPresets = this.vtkColorMaps.rgbPresetNames || [];
        const options = [];

        if (popularFirst) {
            // Add popular colormaps first
            POPULAR_COLORMAPS.forEach(name => {
                if (allPresets.includes(name)) {
                    options.push({ value: name, label: name, isPopular: true });
                }
            });

            // Add separator
            options.push({ value: '', label: '----------', disabled: true });

            // Add remaining colormaps
            allPresets
                .filter(name => !POPULAR_COLORMAPS.includes(name))
                .sort()
                .forEach(name => {
                    options.push({ value: name, label: name, isPopular: false });
                });
        } else {
            allPresets.sort().forEach(name => {
                options.push({ value: name, label: name, isPopular: POPULAR_COLORMAPS.includes(name) });
            });
        }

        return options;
    }

    /**
     * Set auto range mode
     * @param {boolean} auto - Whether to use auto range
     */
    setAutoRange(auto) {
        this.autoRange = auto;
    }

    /**
     * Set manual range
     * @param {number} min - Range minimum
     * @param {number} max - Range maximum
     */
    setManualRange(min, max) {
        this.manualRange = [min, max];
    }

    /**
     * Calculate synchronized range for multiple polydatas
     * @param {Object[]} polydatas - Array of VTK polydata objects
     * @param {string} fieldName - Field name
     * @returns {number[]} Combined range [min, max]
     */
    calculateSynchronizedRange(polydatas, fieldName) {
        let globalMin = Infinity;
        let globalMax = -Infinity;

        const isCellData = fieldName.startsWith('cell:');
        const actualFieldName = isCellData
            ? fieldName.substring(5)
            : fieldName.startsWith('point:')
                ? fieldName.substring(6)
                : fieldName;

        polydatas.forEach(polyData => {
            if (!polyData) return;

            const data = isCellData ? polyData.getCellData() : polyData.getPointData();
            const scalarArray = data?.getArrayByName(actualFieldName);

            if (scalarArray) {
                const range = scalarArray.getRange();
                globalMin = Math.min(globalMin, range[0]);
                globalMax = Math.max(globalMax, range[1]);
            }
        });

        if (globalMin === Infinity || globalMax === -Infinity) {
            return [0, 1]; // Default range
        }

        return [globalMin, globalMax];
    }

    /**
     * Find common fields between multiple polydatas
     * @param {Object[]} polydatas - Array of VTK polydata objects
     * @returns {Object} Object with commonPointFields, commonCellFields, commonAllFields
     */
    findCommonFields(polydatas) {
        if (!polydatas || polydatas.length === 0) {
            return { commonPointFields: [], commonCellFields: [], commonAllFields: [] };
        }

        // Get fields from first polydata
        const firstFields = this.extractFields(polydatas[0]);
        let commonPointFields = firstFields.pointFields.map(f => f.name);
        let commonCellFields = firstFields.cellFields.map(f => f.name);

        // Intersect with remaining polydatas
        for (let i = 1; i < polydatas.length; i++) {
            const fields = this.extractFields(polydatas[i]);
            const pointNames = fields.pointFields.map(f => f.name);
            const cellNames = fields.cellFields.map(f => f.name);

            commonPointFields = commonPointFields.filter(n => pointNames.includes(n));
            commonCellFields = commonCellFields.filter(n => cellNames.includes(n));
        }

        return {
            commonPointFields,
            commonCellFields,
            commonAllFields: [...commonPointFields, ...commonCellFields]
        };
    }
}

export default FieldVisualization;
