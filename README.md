# ComfyUI-GeomPack

Geometry processing custom nodes for ComfyUI, powered by [trimesh](https://trimesh.org/), [libigl](https://libigl.github.io/), and [CGAL](https://www.cgal.org/).

## Overview

ComfyUI-GeomPack provides a comprehensive set of mesh processing nodes for ComfyUI, combining the power of multiple geometry processing libraries. Perfect for mesh cleanup, optimization, remeshing, analysis, and transformation workflows.

## Features

- **Mesh I/O**: Load and save OBJ, PLY, STL, OFF, and more formats via trimesh
- **Dual 3D Preview**: Two interactive viewers - Three.js for general use, VTK.js for scientific visualization
- **Mesh Analysis**: Display statistics, compute normals, analyze geometry properties
- **Mesh Transforms**: Center meshes at origin with one click
- **CGAL Remeshing**: Isotropic remeshing for uniform triangle quality (via PyMeshLab)
- **Blender Integration**: UV unwrapping and remeshing via Blender subprocess
- **Primitive Creation**: Generate cubes, spheres, and planes
- **Trimesh Integration**: Enhanced mesh handling with automatic validation
- **High-poly Support**: Optimized for meshes with 100K+ faces

## Installation

### Method 1: Manual Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-GeomPack.git
```

2. Install dependencies:
```bash
cd ComfyUI-GeomPack
pip install -r requirements.txt
```

3. Install Blender (optional, required for Blender nodes):
   - macOS: `brew install --cask blender` or download from https://www.blender.org/download/
   - Linux: `sudo apt install blender` or `sudo snap install blender`
   - Windows: Download from https://www.blender.org/download/

4. Restart ComfyUI

### Method 2: ComfyUI Manager (Coming Soon)

Install via ComfyUI Manager once published to the registry.

## Quick Start

### Basic Workflow
1. **Create a mesh**: Use the `Create Primitive` node to generate a cube, sphere, or plane
2. **View info**: Connect to `Mesh Info` node to see detailed statistics
3. **Apply remeshing**: Use `CGAL Remesh` to create uniform triangle meshes
4. **Save**: Export with `Save Mesh` node

### CGAL Remeshing Workflow
```
Create Primitive (cube) → Mesh Info → CGAL Remesh → Mesh Info → Save Mesh
```

**Result**: Converts a simple 8-vertex cube into a uniformly remeshed 386-vertex mesh with consistent triangle sizes!

## Available Nodes

### geompack/io
- **Load Mesh**: Load meshes from file (OBJ, PLY, STL, OFF, etc.)
- **Save Mesh**: Save meshes to file with trimesh export

### geompack/analysis
- **Mesh Info**: Display detailed statistics including:
  - Vertex/face/edge counts
  - Volume and surface area
  - Watertight status
  - Bounding box and extents
  - Metadata

### geompack/analysis/distance
- **Hausdorff Distance**: Compute Hausdorff distance between two meshes
  - Measures maximum distance between point sets
  - Returns symmetric and one-sided distances
  - Configurable sample count
  - Useful for measuring worst-case deviation
- **Chamfer Distance**: Compute Chamfer distance between two meshes
  - Average of squared nearest neighbor distances
  - More sensitive to overall shape similarity
  - Configurable sample count
  - Useful for shape matching
- **Compute SDF**: Generate Signed Distance Field for a mesh
  - Voxel-based representation
  - Negative values = inside, positive = outside
  - Configurable resolution (16-256³)
  - Useful for occupancy queries

### geompack/conversion
- **Mesh to Point Cloud**: Sample points from mesh surface
  - Multiple sampling methods (uniform, even, face_weighted)
  - Optional normal computation
  - Configurable sample count
  - Returns POINT_CLOUD data type

### geompack/primitives
- **Create Primitive**: Generate geometric primitives
  - Cube (8 vertices, 12 faces)
  - Sphere (icosphere with subdivision levels)
  - Plane (subdivided grid)

### geompack/pymeshlab
- **PyMeshLab Remesh (Isotropic)**: Isotropic remeshing algorithm
  - Creates uniform triangle sizes
  - Target edge length control
  - Iteration count parameter
  - Preserves volume
  - Fast, no Blender dependency

### geompack/transforms
- **Center Mesh**: Center mesh at origin (0, 0, 0)
  - Uses bounding box center
  - Simple one-click centering
  - Preserves mesh metadata

### geompack/visualization
- **Preview Mesh (3D)**: Interactive 3D mesh viewer
  - Powered by Three.js
  - Orbit controls (rotate, pan, zoom)
  - Real-time preview directly in ComfyUI
  - No need to save files first!
- **Preview Mesh (VTK)**: Scientific visualization viewer
  - Powered by VTK.js
  - Trackball camera controls
  - Better for scientific visualization and analysis
  - Excellent for large meshes and datasets

### geompack/uv
- **xAtlas UV Unwrap**: Fast UV unwrapping using xatlas
  - No Blender dependency
  - Optimized for lightmaps and texture atlasing
  - Same algorithm as Blender 3.6+
  - Production-ready quality
- **libigl LSCM Unwrap**: Conformal UV mapping
  - Least Squares Conformal Maps
  - Angle-preserving parameterization
  - Fast, direct solution
  - No Blender dependency
- **Blender Cube Projection**: Cube projection UV mapping
  - Perfect for box-like geometry
  - Projects onto 6 cube faces
  - Configurable cube size
- **Blender Cylinder Projection**: Cylinder projection UV mapping
  - Perfect for cylindrical objects (bottles, pipes)
  - Configurable radius
  - Equirectangular projection
- **Blender Sphere Projection**: Sphere projection UV mapping
  - Perfect for spherical objects (planets, eyes)
  - Equirectangular projection
  - Single seam
- **Blender UV Unwrap**: Smart UV unwrapping using Blender
  - Angle-based seam detection
  - Configurable island margin
  - Exports mesh with UV coordinates

### geompack/blender
- **Blender Voxel Remesh**: Voxel-based remeshing
  - Creates uniform, watertight meshes
  - Configurable voxel size
  - Good for mesh repair
- **Blender Quadriflow Remesh**: Quad-dominant remeshing
  - Field-aligned quad mesh generation
  - Target face count control
  - Better for animation and subdivision

### geompack/examples
- **Example Node**: Template for creating custom nodes

## Mesh Data Format

Meshes are now represented as **`trimesh.Trimesh`** objects, providing:

```python
mesh = trimesh.Trimesh(vertices=np.ndarray, faces=np.ndarray)

# Built-in properties:
mesh.vertices          # (N, 3) numpy array of XYZ coordinates
mesh.faces             # (F, 3) numpy array of triangle indices
mesh.vertex_normals    # (N, 3) per-vertex normals (auto-computed)
mesh.bounds            # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
mesh.centroid          # [x, y, z] center of mass
mesh.volume            # Volume of the mesh (if watertight)
mesh.area              # Total surface area
mesh.is_watertight     # Boolean: mesh is closed?
mesh.metadata          # Dictionary for custom data

# Methods:
mesh.export('output.obj')    # Save to file
mesh.apply_transform(matrix) # Transform mesh
# ... and many more!
```

## Example Workflows

### CGAL Isotropic Remeshing
```
Create Primitive (cube, size=1.0)
  → CGAL Remesh (target_edge_length=0.15, iterations=3)
  → Mesh Info
  → Save Mesh (output.obj)
```

**Input**: 8 vertices, 12 faces
**Output**: 386 vertices, 768 faces with uniform triangle sizes
**Volume Preserved**: 1.000000 (0.00% error)

### Mesh Loading and Analysis
```
Load Mesh (input.obj)
  → Mesh Info (view statistics)
  → CGAL Remesh (clean up mesh)
  → Save Mesh (output.obj)
```

### Primitive Creation
```
Create Primitive (sphere, subdivisions=3)
  → Mesh Info (check watertight status)
  → Save Mesh (sphere.stl)
```

## Development Status

**Current Version**: 0.2.0

### Recently Completed
- ✅ Migrated from dict-based meshes to trimesh.Trimesh objects
- ✅ Load Mesh (with fallback to libigl)
- ✅ Save Mesh (all trimesh-supported formats)
- ✅ Create Primitive (cube, sphere, plane)
- ✅ Mesh Info (enhanced with volume, watertight detection)
- ✅ PyMeshLab Isotropic Remeshing
- ✅ Blender UV Unwrapping (Smart UV Project)
- ✅ Blender Voxel Remeshing
- ✅ Blender Quadriflow Remeshing
- ✅ Center Mesh (transform to origin)
- ✅ Preview Mesh (3D) - Interactive Three.js viewer widget!
- ✅ Preview Mesh (VTK) - Scientific visualization with VTK.js with controls and mesh info panel!
- ✅ Distance Metrics - Hausdorff, Chamfer, and SDF computation
- ✅ Mesh to Point Cloud conversion with multiple sampling methods
- ✅ xAtlas UV Unwrap - Fast, production-ready UV unwrapping
- ✅ Blender UV Projections - Cube, Cylinder, and Sphere projections
- ✅ libigl LSCM - Conformal UV mapping

### Coming Soon (Future Phases)
- [ ] Mesh Decimation
- [ ] Mesh Subdivision
- [ ] Mesh Scaling/Rotation nodes
- [ ] Laplacian Smoothing
- [ ] Boolean Operations (CGAL)
- [ ] Mesh Repair
- [ ] PyNanoInstantMeshes integration
- [ ] Native CGAL remeshing (when Python binding issues resolved)

See [MVP_PLAN.md](MVP_PLAN.md) for the complete roadmap.

## Requirements

- Python 3.8+
- ComfyUI
- **trimesh** 3.15.0+ (core mesh handling)
- **pymeshlab** 2022.2+ (isotropic remeshing)
- **libigl** 2.6.1+ (LSCM UV unwrapping)
- **numpy** 1.21.0+
- **scipy** 1.7.0+
- **matplotlib** 3.5.0+
- **Pillow** 9.0.0+
- **point-cloud-utils** 0.30.0+ (distance metrics)
- **mesh-to-sdf** 0.0.14+ (SDF computation)
- **xatlas** 0.0.11+ (fast UV unwrapping)
- **Blender** (optional, for Blender-based UV and remeshing nodes)

## Technical Details

### CGAL Isotropic Remeshing

The `CGAL Remesh` node implements the incremental triangle-based isotropic remeshing algorithm from:

> Botsch, M., & Kobbelt, L. (2004). "A remeshing approach to multiresolution modeling."
> Eurographics Symposium on Geometry Processing.

**How it works:**
1. Converts trimesh to CGAL Polyhedron_3
2. Applies isotropic remeshing with:
   - **Edge splits**: Edges longer than target length
   - **Edge collapses**: Edges shorter than target length
   - **Edge flips**: Improve triangle quality
   - **Tangential smoothing**: Relax vertex positions
3. Converts back to trimesh with metadata preservation

**Parameters:**
- `target_edge_length`: Desired edge length for triangles (smaller = denser mesh)
- `iterations`: Number of refinement passes (typically 3-10)

**Benefits:**
- More uniform triangle distribution
- Better numerical stability for simulations
- Improved mesh quality for rendering
- Volume preservation

### Blender Integration

The Blender nodes integrate Blender's powerful mesh processing capabilities via subprocess:

**How it works:**
1. Mesh is exported to temporary OBJ file
2. Blender is launched in background mode (`--background`)
3. Python script is executed via `--python-expr`
4. Blender performs the operation (UV unwrap, remesh, etc.)
5. Result is exported to temporary OBJ file
6. Result is loaded back into trimesh

**Available Blender Operations:**

**UV Unwrapping (Smart UV Project):**
- Automatically creates UV seams based on angle threshold
- Configurable island margin for texture bleeding prevention
- Preserves mesh geometry, adds UV coordinates

**Voxel Remeshing:**
- Voxelizes the input mesh at specified resolution
- Reconstructs surface from voxel grid
- Creates watertight, uniform meshes
- Good for mesh repair and cleanup

**Quadriflow Remeshing:**
- Field-aligned quad mesh generation
- Better topology for animation and subdivision
- Target face count control
- Outputs quad-dominant mesh (triangulated by trimesh)

**Note:** Blender must be installed and accessible in your PATH for these nodes to work.

### 3D Preview Widgets

ComfyUI-GeomPack provides two interactive 3D viewers directly in ComfyUI:

#### Preview Mesh (3D) - Three.js Viewer

The Three.js viewer provides a general-purpose 3D visualization with excellent rendering quality:

**Features:**
- **Interactive Camera:** Orbit, pan, and zoom with mouse/trackpad controls
- **Real-time Rendering:** Smooth 60fps rendering with WebGL
- **Automatic Scaling:** Camera automatically adjusts to fit mesh
- **Professional Lighting:** Three-point lighting setup for clear visualization
- **Grid and Axes:** Visual reference for orientation

**How it works:**
1. Mesh is exported to GLB format (optimized for web)
2. File is saved to ComfyUI's output directory
3. Three.js viewer loads and displays the mesh
4. OrbitControls enable interactive exploration

**Controls:**
- **Left Mouse:** Rotate camera around mesh
- **Right Mouse / Two-finger drag:** Pan camera
- **Scroll / Pinch:** Zoom in/out
- **Automatic:** Camera resets to fit mesh on load

**Technical Details:**
- Uses Three.js v0.160.0 from CDN (no local dependencies)
- Supports GLB and OBJ formats
- 512x512px viewport (customizable in code)
- Renders in ComfyUI's output directory (accessible via /view API)

#### Preview Mesh (VTK) - VTK.js Viewer

The VTK.js viewer is optimized for scientific visualization and large datasets:

**Features:**
- **Trackball Controls:** Professional scientific visualization camera controls
- **Scientific Rendering:** Optimized for mesh analysis and inspection
- **High Performance:** Excellent for large meshes (100K+ vertices)
- **Clean Interface:** Focused on geometry visualization without distractions

**How it works:**
1. Mesh is exported to STL format (native VTK.js format)
2. File is saved to ComfyUI's output directory
3. VTK.js viewer loads and displays the mesh
4. TrackballCamera enables precise exploration

**Controls:**
- **Left Mouse:** Rotate camera (trackball mode)
- **Middle Mouse / Shift + Left:** Pan camera
- **Right Mouse / Scroll:** Zoom in/out
- **Automatic:** Camera resets to fit mesh on load

**Technical Details:**
- Uses VTK.js v29.5.0 from Skypack CDN (no local dependencies)
- Supports STL and OBJ formats
- 512x512px viewport (customizable in code)
- Renders in ComfyUI's output directory (accessible via /view API)

**When to use which viewer:**
- **Three.js**: General 3D visualization, artistic rendering, better lighting
- **VTK.js**: Scientific analysis, large datasets, mesh inspection, precise measurements

**No External Tools Required:** Unlike other 3D preview solutions, both widgets render directly in the ComfyUI interface - no need to save files or use external viewers!

## Testing

Run the test suite to verify installation:

```bash
python test_nodes.py
```

Expected output:
```
============================================================
🎉 ALL TESTS PASSED!
============================================================

The following nodes are working correctly:
  ✓ CreatePrimitive (cube, sphere, plane)
  ✓ SaveMesh (OBJ format)
  ✓ LoadMesh (OBJ format)
  ✓ MeshInfoNode (with trimesh enhancements)
  ✓ CGALRemeshNode (isotropic remeshing)
```

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_nodes.py`
5. Submit a pull request

### Adding New Nodes

Follow the pattern in `nodes.py`:

```python
class MyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),  # trimesh.Trimesh object
                "parameter": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "process"
    CATEGORY = "geompack/custom"

    def process(self, mesh, parameter):
        # Your processing logic using trimesh API
        # ...
        return (processed_mesh,)
```

## Planned Features

The following features are planned for future releases:

### Point Cloud Visualization
- **plas.io Integration**: Advanced point cloud viewer for LAS/LAZ files
  - Support for massive point cloud datasets
  - Intensity and RGB visualization
  - Fast WebGL-based rendering
  - Interactive classification tools

### VTK Viewer Enhancements
- **Interactive Controls** (✅ Implemented)
  - Toggle edges on/off
  - Switch between surface/wireframe/points rendering modes
  - Screenshot capture and save
  - Reset camera view
- **Mesh Information Panel** (✅ Implemented)
  - Display vertex and face counts
  - Show bounding box and extents
  - Report watertight status
  - Calculate volume and surface area
  - List scalar field names
- **Scalar Field Visualization** (In Progress)
  - Dropdown selector for different fields
  - Color mapping based on field values
  - Adjustable color transfer functions
  - Scalar bar legend

### Additional 3D Viewers
- **Gaussian Splatting**: GaussianSplats3D for neural radiance field visualization
- **model-viewer**: glTF/GLB viewer with AR support

See the [Development Status](#development-status) section for the complete roadmap.

## License

MIT License - See LICENSE file for details

## Credits

- Built on [trimesh](https://trimesh.org/) by Michael Dawson-Haggerty
- CGAL integration via [CGAL Python bindings](https://github.com/CGAL/cgal-swig-bindings)
- [libigl](https://libigl.github.io/) by Alec Jacobson and others
- Inspired by existing ComfyUI 3D tools and workflows

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join the discussion in GitHub Discussions
- **Documentation**: See the [wiki](https://github.com/yourusername/ComfyUI-GeomPack/wiki) for detailed guides

## Changelog

### Version 0.2.0 (Current)
- **Major Refactor**: Migrated from dictionary-based meshes to trimesh.Trimesh objects
- **Added**: CGAL isotropic remeshing node
- **Added**: Primitive creation (cube, sphere, plane)
- **Enhanced**: Mesh Info now shows volume, watertight status, and more
- **Improved**: File I/O with trimesh (supports more formats)
- **Updated**: All nodes now use trimesh API
- **Added**: Comprehensive test suite

### Version 0.1.0
- Initial MVP release
- Basic node structure and boilerplate
- Example nodes demonstrating functionality
