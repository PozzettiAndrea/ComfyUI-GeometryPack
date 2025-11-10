# ComfyUI-GeometryPack Test Suite

Comprehensive test suite for all GeometryPack nodes with visual output verification.

## Example Test Results

### Stanford Bunny Processing
![Stanford Bunny Example](../assets/stanford_bunny.png)

### Remeshing Operations
![Remeshing Example](../assets/remeshing.png)

Each test generates its own output folder with before/after meshes and rendered images for visual verification.

---

## Directory Structure

```
tests/
├── test_scripts/          # All test files and fixtures
│   ├── conftest.py       # Pytest fixtures and helpers
│   ├── test_*.py         # Individual test modules
│   └── .gitignore
├── outputs/              # Generated test outputs
│   ├── meshes/          # Saved mesh files (.obj, .ply, .stl)
│   └── renders/         # PNG preview images
├── assets/              # Test input files (bunny.stl, etc.)
└── README.md            # This file
```

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all unit tests
pytest -m "unit" -v

# Run specific test file
pytest test_scripts/test_primitives.py -v

# Run with output images
pytest -v
```

## Test Organization

### Unit Tests (Fast, No Optional Dependencies)

**test_io.py** - I/O operations
- `test_load_mesh_stl` - Load STL file (Stanford Bunny)
- `test_save_mesh_obj` - Save mesh to OBJ format
- `test_save_mesh_ply` - Save mesh to PLY format
- `test_save_load_cycle` - Round-trip save/load for all formats (OBJ, PLY, STL, OFF)

**test_primitives.py** - Primitive generation
- `test_create_primitive_shapes` - Create cube, plane primitives
- `test_create_sphere` - Create sphere and verify radius
- `test_create_primitive_with_subdivisions` - Test subdivision levels
- `test_create_primitive_sizes` - Test different primitive sizes

**test_analysis.py** - Mesh analysis
- `test_mesh_info_cube` - Get mesh statistics for cube
- `test_mesh_info_bunny` - Get mesh statistics for Stanford Bunny
- `test_mark_boundary_edges_closed_mesh` - Boundary detection on closed mesh
- `test_mark_boundary_edges_open_mesh` - Boundary detection on open mesh
- `test_mesh_info_volume_area` - Verify volume/area calculations

**test_transforms.py** - Transformations
- `test_center_mesh` - Center mesh at origin
- `test_center_mesh_preserves_shape` - Verify shape preservation

**test_conversion.py** - Format conversions
- `test_strip_mesh_adjacency` - Convert mesh to point cloud (no sampling)
- `test_strip_mesh_with_normals` - Point cloud with normals
- `test_mesh_to_point_cloud_sampling` - Sample point clouds (100, 500, 1000 points)
- `test_mesh_to_point_cloud_methods` - Test uniform/even/face_weighted sampling
- `test_point_cloud_with_normals` - Generate point clouds with normals

**test_repair.py** - Mesh repair
- `test_fix_normals` - Fix inconsistent normals
- `test_check_normals` - Check normal consistency
- `test_fill_holes` - Fill holes in open mesh
- `test_compute_normals_faceted` - Compute faceted normals
- `test_compute_normals_smooth` - Compute smooth vertex normals
- `test_visualize_normal_field` - Generate normal scalar fields

**test_remeshing.py** - Remeshing operations
- `test_mesh_decimation` - Reduce triangle count
- `test_mesh_subdivision` - Subdivide mesh (1, 2, 3 iterations)
- `test_laplacian_smoothing` - Laplacian smoothing

**test_visualization.py** - Visualization exports
- `test_preview_mesh_threejs` - Three.js preview (GLB export)
- `test_preview_mesh_vtk` - VTK.js preview (STL export)
- `test_preview_mesh_vtk_filters` - VTK.js with filters
- `test_preview_mesh_vtk_fields` - VTK.js with scalar fields

### Optional Dependency Tests

**test_remeshing.py** - Advanced remeshing
- `test_pymeshlab_remesh` - PyMeshLab isotropic remeshing (requires pymeshlab)
- `test_cgal_remesh` - CGAL isotropic remeshing (requires cgal)
- `test_instant_meshes_remesh` - InstantMeshes remeshing (requires PyNanoInstantMeshes)

**test_uv.py** - UV unwrapping
- `test_xatlas_unwrap` - xAtlas UV unwrapping (requires xatlas)
- `test_libigl_lscm` - libigl LSCM unwrapping (requires igl)
- `test_libigl_harmonic` - libigl harmonic unwrapping (requires igl)
- `test_libigl_arap` - libigl ARAP unwrapping (requires igl)

**test_distance.py** - Distance metrics
- `test_hausdorff_distance_identical` - Hausdorff distance (identical meshes)
- `test_hausdorff_distance_different` - Hausdorff distance (different meshes)
- `test_chamfer_distance_identical` - Chamfer distance (identical meshes)
- `test_chamfer_distance_different` - Chamfer distance (different meshes)
- `test_compute_sdf` - Signed distance field generation (32, 64 resolution)

### Blender Tests (Slow)

**test_remeshing.py** - Blender remeshing
- `test_blender_voxel_remesh` - Blender voxel remeshing
- `test_blender_quadriflow_remesh` - Blender Quadriflow remeshing

**test_uv.py** - Blender UV unwrapping
- `test_blender_uv_unwrap` - Blender Smart UV Project
- `test_blender_cube_projection` - Cube projection UV
- `test_blender_cylinder_projection` - Cylinder projection UV
- `test_blender_sphere_projection` - Sphere projection UV

### Integration Tests

**test_integration.py** - Multi-node pipelines
- `test_pipeline_create_analyze_save` - Create → Analyze → Save
- `test_pipeline_load_remesh_save` - Load → Remesh → Save
- `test_pipeline_create_subdivide_smooth` - Create → Subdivide → Smooth
- `test_pipeline_repair_analyze` - Repair → Analyze
- `test_pipeline_mesh_to_pointcloud` - Mesh → Point Cloud conversion
- `test_pipeline_boundary_detection_visualization` - Boundary → Visualization
- `test_pipeline_full_processing` - Full processing pipeline

## Running Specific Test Categories

```bash
# Unit tests only (fastest)
pytest tests/ -m "unit"

# Optional dependency tests
pytest tests/ -m "optional"

# Blender tests (requires Blender installed)
pytest tests/ -m "blender"

# Slow tests
pytest tests/ -m "slow"

# Skip slow tests
pytest tests/ -m "not slow"
```

## Test Outputs

All tests generate visual outputs for manual verification:

- **Meshes**: `tests/outputs/meshes/` - OBJ, PLY, STL files
- **Renders**: `tests/outputs/renders/` - PNG preview images

## Coverage

Run tests with coverage report:

```bash
pytest tests/ --cov=nodes --cov-report=html
open htmlcov/index.html
```

## CI/CD

GitHub Actions runs tests automatically on:
- Ubuntu, macOS, Windows
- Python 3.10
- Uploads test outputs as artifacts

## Test Statistics

- **Total test files**: 10
- **Total test cases**: 60+
- **Node coverage**: 36/36 nodes tested
- **All tests run on CPU** - No GPU required
