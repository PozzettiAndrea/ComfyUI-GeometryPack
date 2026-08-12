/**
 * OBJLoader - OBJ mesh file loader
 *
 * Loads Wavefront OBJ files using VTK.js vtkOBJReader.
 * Includes normals computation for proper shading.
 */

/**
 * Load an OBJ file and return polydata
 * @param {ArrayBuffer} arrayBuffer - File data as ArrayBuffer
 * @param {Object} vtk - VTK.js global object
 * @returns {Object} Result object with polyData and metadata
 */
export async function loadOBJ(arrayBuffer, vtk) {
    const vtkOBJReader = vtk.IO.Misc.vtkOBJReader;
    const vtkPolyDataNormals = vtk.Filters.Core.vtkPolyDataNormals;

    const reader = vtkOBJReader.newInstance();

    // OBJ files are text-based
    const text = new TextDecoder().decode(arrayBuffer);
    reader.parseAsText(text);

    const rawPolyData = reader.getOutputData();
    if (!rawPolyData) {
        throw new Error('Failed to parse OBJ data');
    }

    // Apply normals filter for proper shading
    const normalsFilter = vtkPolyDataNormals.newInstance();
    normalsFilter.setInputConnection(reader.getOutputPort());
    normalsFilter.update();

    const polyData = normalsFilter.getOutputData();

    return {
        polyData,
        reader,
        normalsFilter,
        outputPort: normalsFilter.getOutputPort(),
        metadata: {
            format: 'OBJ',
            numPoints: polyData.getNumberOfPoints(),
            numPolys: polyData.getNumberOfPolys(),
            bounds: polyData.getBounds()
        }
    };
}

/**
 * Check if a filepath is an OBJ file
 * @param {string} filepath - File path or URL
 * @returns {boolean}
 */
export function canLoadOBJ(filepath) {
    return filepath.toLowerCase().includes('.obj');
}

export default {
    load: loadOBJ,
    canLoad: canLoadOBJ,
    format: 'OBJ'
};
