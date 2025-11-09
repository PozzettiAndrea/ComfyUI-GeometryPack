# Blender Integration Guide

ComfyUI-GeomPack integrates Blender's powerful mesh processing capabilities via subprocess calls. This allows you to leverage Blender's UV unwrapping and remeshing tools directly in your ComfyUI workflows.

## Overview

Three Blender nodes are available:
1. **Blender UV Unwrap** - Smart UV unwrapping
2. **Blender Voxel Remesh** - Voxel-based remeshing
3. **Blender Quadriflow Remesh** - Quad-dominant remeshing

## Installation

### Install Blender

**macOS:**
```bash
brew install --cask blender
```

**Linux:**
```bash
sudo apt install blender
# or
sudo snap install blender
```

**Windows:**
Download from https://www.blender.org/download/

### Verify Installation

```bash
blender --version
```

You should see output like:
```
Blender 4.5.4 LTS
```

## How It Works

The integration uses a subprocess pattern:

1. **Export**: Mesh is exported to a temporary OBJ file
2. **Launch**: Blender is launched in background mode (`--background`)
3. **Execute**: Python script is executed via `--python-expr`
4. **Process**: Blender performs the operation
5. **Export**: Result is exported to a temporary OBJ file
6. **Import**: Result is loaded back into trimesh
7. **Cleanup**: Temporary files are deleted

This approach ensures:
- No Blender GUI is shown
- No manual intervention required
- Fast and automated processing
- Full integration with ComfyUI workflows

## Node Documentation

### 1. Blender UV Unwrap

**Category**: `geompack/blender`

**Description**: Generates UV coordinates for texturing using Blender's Smart UV Project algorithm.

**Parameters**:
- `mesh` (MESH): Input mesh
- `angle_limit` (FLOAT): Angle threshold for creating seams in degrees (default: 66.0, range: 1.0-89.0)
- `island_margin` (FLOAT): Spacing between UV islands (default: 0.02, range: 0.0-1.0)

**Returns**:
- `unwrapped_mesh` (MESH): Mesh with UV coordinates

**Use Cases**:
- Preparing meshes for texture mapping
- Creating UV layouts for baking
- Automatic seam detection

**Example**:
```
Load Mesh → Blender UV Unwrap (angle_limit=66.0, island_margin=0.02) → Save Mesh
```

**Technical Details**:
- Uses `bpy.ops.uv.smart_project()`
- Automatically creates seams based on angle threshold
- Islands are separated by configurable margin
- UV coordinates are exported in the OBJ file
- Preserves original mesh geometry

**Test Results**:
```
Input:  8 vertices, 12 faces (cube)
Output: 24 vertices, 12 faces (with UV coordinates)
```

---

### 2. Blender Voxel Remesh

**Category**: `geompack/blender`

**Description**: Creates uniform, watertight meshes by voxelizing the input and reconstructing the surface.

**Parameters**:
- `mesh` (MESH): Input mesh
- `voxel_size` (FLOAT): Voxel size for remeshing (default: 0.05, range: 0.001-1.0)
  - Smaller values = higher resolution = more faces
  - Larger values = lower resolution = fewer faces

**Returns**:
- `remeshed_mesh` (MESH): Voxel-remeshed mesh

**Use Cases**:
- Mesh repair and cleanup
- Creating watertight meshes for 3D printing
- Simplifying complex meshes
- Uniform mesh density

**Example**:
```
Load Mesh → Blender Voxel Remesh (voxel_size=0.1) → Save Mesh
```

**Technical Details**:
- Uses `bpy.ops.object.voxel_remesh()`
- Converts mesh to voxel grid at specified resolution
- Reconstructs surface using marching cubes-like algorithm
- Always produces watertight meshes
- Good for fixing non-manifold geometry

**Test Results**:
```
Input:  42 vertices, 80 faces (sphere, subdivisions=1)
Output: 2005 vertices, 1020 faces (voxel_size=0.1)
```

**Performance Notes**:
- Processing time depends on voxel size
- Smaller voxel sizes = longer processing time
- Typical range: 0.01 to 0.2 for most meshes

---

### 3. Blender Quadriflow Remesh

**Category**: `geompack/blender`

**Description**: Creates quad-dominant meshes with field-aligned topology, better for animation and subdivision.

**Parameters**:
- `mesh` (MESH): Input mesh
- `target_face_count` (INT): Target number of faces in output (default: 5000, range: 100-100000)

**Returns**:
- `remeshed_mesh` (MESH): Quad-remeshed mesh (triangulated by trimesh)

**Use Cases**:
- Creating animation-ready meshes
- Preparing for subdivision surface modeling
- Retopology of high-resolution scans
- Better edge flow for character models

**Example**:
```
Load Mesh → Blender Quadriflow Remesh (target_face_count=5000) → Save Mesh
```

**Technical Details**:
- Uses `bpy.ops.object.quadriflow_remesh()`
- Field-aligned quad mesh generation
- Optimized for subdivision surfaces
- Output is quad-dominant (may contain some triangles)
- Trimesh automatically triangulates quads for compatibility

**Test Results**:
```
Input:  42 vertices, 80 faces (sphere, subdivisions=1)
Output: 2036 vertices, 1022 faces (target_face_count=500)
```

**Notes**:
- Actual face count may differ from target
- Quadriflow algorithm optimizes for good topology
- Better edge flow than isotropic remeshing

---

## Comparison: Remeshing Algorithms

| Algorithm | Output | Use Case | Speed |
|-----------|--------|----------|-------|
| **CGAL Isotropic** (PyMeshLab) | Uniform triangles | Simulation, analysis | Fast |
| **Blender Voxel** | Watertight, uniform | Repair, 3D printing | Medium |
| **Blender Quadriflow** | Quad-dominant | Animation, subdivision | Slow |

**When to use each:**

- **CGAL Isotropic**: When you need uniform triangle sizes for FEM simulation or mesh analysis
- **Blender Voxel**: When you need to repair broken meshes or create watertight geometry
- **Blender Quadriflow**: When you need good topology for animation, rigging, or subdivision surfaces

## Troubleshooting

### "Blender not found" Error

**Problem**: Node fails with `RuntimeError: Blender not found`

**Solution**:
1. Verify Blender is installed: `blender --version`
2. Add Blender to PATH:
   - macOS: Already in PATH if installed via Homebrew
   - Linux: Should be in `/usr/bin/blender` or `/usr/local/bin/blender`
   - Windows: Add Blender installation folder to PATH

### Empty Output Mesh

**Problem**: Remeshing produces 0 vertices, 0 faces

**Solution**:
- Check Blender version compatibility (tested with Blender 4.5.4 LTS)
- Try different parameter values (e.g., larger voxel size, different face count)
- Check input mesh is valid (use Mesh Info node)

### Slow Performance

**Problem**: Remeshing takes a long time

**Solution**:
- **Voxel Remesh**: Increase `voxel_size` (e.g., 0.1 instead of 0.01)
- **Quadriflow**: Reduce `target_face_count` (e.g., 1000 instead of 10000)
- For large meshes, consider decimating first

### Subprocess Timeout

**Problem**: Operation times out after 5 minutes

**Solution**:
- Edit the `timeout=300` parameter in the node code to increase timeout
- Simplify the input mesh before processing
- Use coarser parameters (larger voxel size, lower face count)

## Advanced Usage

### Custom Blender Scripts

You can modify the Blender scripts in `nodes.py` to add more Blender operations:

```python
script = f"""
import bpy

# Your custom Blender operations here
# Example: Apply a modifier
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.ops.object.modifier_apply()
"""
```

### Batch Processing

Combine multiple Blender operations in a single workflow:

```
Load Mesh
  → Blender Voxel Remesh (repair mesh)
  → Blender Quadriflow Remesh (retopology)
  → Blender UV Unwrap (add UVs)
  → Save Mesh
```

## Technical Implementation

### Code Structure

```python
def _find_blender():
    """Find Blender executable on system"""
    # Checks common paths: /Applications/Blender.app, /usr/bin/blender, etc.

class BlenderUVUnwrapNode:
    def uv_unwrap(self, mesh, angle_limit, island_margin):
        # 1. Find Blender
        blender_path = _find_blender()

        # 2. Export mesh to temp OBJ
        input_path = tempfile.mktemp(suffix='.obj')
        mesh.export(input_path)

        # 3. Create Blender script
        script = f"""
import bpy
bpy.ops.wm.obj_import(filepath='{input_path}')
bpy.ops.uv.smart_project(...)
bpy.ops.wm.obj_export(filepath='{output_path}')
"""

        # 4. Run Blender
        subprocess.run([blender_path, '--background', '--python-expr', script])

        # 5. Load result
        unwrapped = trimesh.load(output_path)

        # 6. Cleanup
        os.unlink(input_path)
        os.unlink(output_path)

        return (unwrapped,)
```

### Tested Blender Versions

- **Blender 4.5.4 LTS** ✓ Fully tested
- Older versions should work but may have different API parameters

### Compatibility Notes

- **UV Unwrap**: Compatible with all Blender 3.x and 4.x versions
- **Voxel Remesh**: Available in Blender 2.81+
- **Quadriflow**: Available in Blender 2.81+
  - Note: Parameter compatibility varies by version
  - This implementation uses minimal parameter set for maximum compatibility

## Examples

### Example 1: UV Unwrap for Texturing

**Goal**: Prepare a mesh for texture painting

**Workflow**:
```
Load Mesh (character.obj)
  → Mesh Info (check stats)
  → Blender UV Unwrap (angle_limit=66.0, island_margin=0.02)
  → Save Mesh (character_unwrapped.obj)
```

**Result**: Mesh with UV coordinates ready for texture painting in Blender, Substance Painter, etc.

---

### Example 2: Mesh Repair for 3D Printing

**Goal**: Fix broken mesh and make it watertight for 3D printing

**Workflow**:
```
Load Mesh (broken_scan.obj)
  → Mesh Info (check watertight status)
  → Blender Voxel Remesh (voxel_size=0.05)
  → Mesh Info (verify watertight=True)
  → Save Mesh (repaired_for_printing.stl)
```

**Result**: Watertight mesh suitable for 3D printing

---

### Example 3: Retopology for Animation

**Goal**: Create animation-ready topology from high-resolution scan

**Workflow**:
```
Load Mesh (scan_highres.obj)
  → Blender Quadriflow Remesh (target_face_count=10000)
  → Blender UV Unwrap (angle_limit=66.0)
  → Save Mesh (character_lowpoly.obj)
```

**Result**: Clean quad-dominant mesh with good edge flow and UV coordinates

---

### Example 4: Multi-Stage Remeshing

**Goal**: Combine multiple remeshing techniques for best results

**Workflow**:
```
Load Mesh (raw_scan.obj)
  → Blender Voxel Remesh (voxel_size=0.1, cleanup)
  → CGAL Remesh (target_edge_length=0.05, uniform triangles)
  → Mesh Info (check quality)
  → Save Mesh (processed.obj)
```

**Result**: Clean, uniform mesh with good quality

## Performance Benchmarks

Tested on MacBook Pro (M1, 16GB RAM):

| Operation | Input | Output | Time |
|-----------|-------|--------|------|
| UV Unwrap | 8v, 12f | 24v, 12f | ~1.5s |
| Voxel Remesh (0.1) | 42v, 80f | 2005v, 1020f | ~2.0s |
| Quadriflow (500 faces) | 42v, 80f | 2036v, 1022f | ~3.5s |

**Notes**:
- Times include Blender startup, processing, and shutdown
- Larger meshes take proportionally longer
- Voxel size and face count significantly affect performance

## Future Improvements

- [ ] Add Blender Decimate modifier
- [ ] Add Blender Smooth modifier
- [ ] Add Blender Shrinkwrap for retopology
- [ ] Support for custom Blender scripts via file input
- [ ] Progress reporting for long operations
- [ ] Parallel Blender processing for batch operations

## Credits

- Blender integration pattern inspired by [ComfyUI-MeshCraft](https://github.com/PozzettiAndrea/ComfyUI-MeshCraft)
- Blender Python API documentation: https://docs.blender.org/api/current/
