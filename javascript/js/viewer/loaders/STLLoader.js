/**
 * STLLoader - STL mesh file loader
 *
 * Loads binary or ASCII STL files using VTK.js vtkSTLReader.
 * Includes normals computation for proper shading.
 */

/**
 * Load an STL file and return polydata
 * @param {ArrayBuffer} arrayBuffer - File data as ArrayBuffer
 * @param {Object} vtk - VTK.js global object
 * @returns {Object} Result object with polyData and metadata
 */
export async function loadSTL(arrayBuffer, vtk) {
    const vtkSTLReader = vtk.IO.Geometry.vtkSTLReader;
    const vtkPolyDataNormals = vtk.Filters.Core.vtkPolyDataNormals;

    const reader = vtkSTLReader.newInstance();
    reader.parseAsArrayBuffer(arrayBuffer);

    const rawPolyData = reader.getOutputData();
    if (!rawPolyData) {
        throw new Error('Failed to parse STL data');
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
            format: 'STL',
            numPoints: polyData.getNumberOfPoints(),
            numPolys: polyData.getNumberOfPolys(),
            bounds: polyData.getBounds()
        }
    };
}

/**
 * Check if a filepath is an STL file
 * @param {string} filepath - File path or URL
 * @returns {boolean}
 */
export function canLoadSTL(filepath) {
    return filepath.toLowerCase().includes('.stl');
}

export default {
    load: loadSTL,
    canLoad: canLoadSTL,
    format: 'STL'
};
