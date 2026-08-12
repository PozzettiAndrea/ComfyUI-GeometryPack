/**
 * VTPLoader - VTK XML PolyData file loader
 *
 * Loads VTP files using VTK.js vtkXMLPolyDataReader.
 * VTP files can contain scalar fields (vertex/face attributes).
 * Includes normals computation for proper shading.
 */

/**
 * Extract field information from polydata
 * @param {Object} polyData - VTK polydata object
 * @returns {Object} Field information
 */
function extractFieldInfo(polyData) {
    const pointData = polyData.getPointData();
    const cellData = polyData.getCellData();

    const pointFields = [];
    const cellFields = [];

    // Extract point (vertex) data arrays
    for (let i = 0; i < pointData.getNumberOfArrays(); i++) {
        const array = pointData.getArray(i);
        const name = array.getName();
        const range = array.getRange();
        pointFields.push({
            name,
            type: 'point',
            numComponents: array.getNumberOfComponents(),
            range: [range[0], range[1]]
        });
    }

    // Extract cell (face) data arrays
    for (let i = 0; i < cellData.getNumberOfArrays(); i++) {
        const array = cellData.getArray(i);
        const name = array.getName();
        const range = array.getRange();
        cellFields.push({
            name,
            type: 'cell',
            numComponents: array.getNumberOfComponents(),
            range: [range[0], range[1]]
        });
    }

    return {
        pointFields,
        cellFields,
        hasFields: pointFields.length > 0 || cellFields.length > 0
    };
}

/**
 * Load a VTP file and return polydata
 * @param {ArrayBuffer} arrayBuffer - File data as ArrayBuffer
 * @param {Object} vtk - VTK.js global object
 * @returns {Object} Result object with polyData, fields, and metadata
 */
export async function loadVTP(arrayBuffer, vtk) {
    const vtkXMLPolyDataReader = vtk.IO.XML.vtkXMLPolyDataReader;
    const vtkPolyDataNormals = vtk.Filters.Core.vtkPolyDataNormals;

    const reader = vtkXMLPolyDataReader.newInstance();
    reader.parseAsArrayBuffer(arrayBuffer);

    const rawPolyData = reader.getOutputData();
    if (!rawPolyData) {
        throw new Error('Failed to parse VTP data');
    }

    // Extract field info BEFORE normals filter (preserves original data)
    const fieldInfo = extractFieldInfo(rawPolyData);

    // Apply normals filter for proper shading
    const normalsFilter = vtkPolyDataNormals.newInstance();
    normalsFilter.setInputConnection(reader.getOutputPort());
    normalsFilter.update();

    const polyData = normalsFilter.getOutputData();

    // Extract field info again from filtered data (should be same)
    const filteredFieldInfo = extractFieldInfo(polyData);

    return {
        polyData,
        reader,
        normalsFilter,
        outputPort: normalsFilter.getOutputPort(),
        fields: filteredFieldInfo,
        metadata: {
            format: 'VTP',
            numPoints: polyData.getNumberOfPoints(),
            numPolys: polyData.getNumberOfPolys(),
            bounds: polyData.getBounds(),
            pointFields: filteredFieldInfo.pointFields.map(f => f.name),
            cellFields: filteredFieldInfo.cellFields.map(f => f.name),
            hasFields: filteredFieldInfo.hasFields
        }
    };
}

/**
 * Check if a filepath is a VTP file
 * @param {string} filepath - File path or URL
 * @returns {boolean}
 */
export function canLoadVTP(filepath) {
    return filepath.toLowerCase().includes('.vtp');
}

export default {
    load: loadVTP,
    canLoad: canLoadVTP,
    format: 'VTP'
};
