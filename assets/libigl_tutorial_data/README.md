# Self-Intersecting Test Meshes (from libigl Tutorial Data)

This directory contains meshes from the **libigl tutorial data** that have been **verified to contain self-intersections** for demonstrating the self-intersection detection and repair nodes in ComfyUI GeometryPack.

## Test Meshes

All meshes are from [alecjacobson/libigl-tutorial-data](https://github.com/alecjacobson/libigl-tutorial-data) and have been verified using libigl CGAL detection:

### 1. **camelhead.off** (749 KB) ✓ Verified
- **Intersections:** 5 pairs, 6 faces affected
- Classic camel head mesh from computer graphics
- Minimal self-intersections - good for basic testing
- Best for: Simple repair demonstrations

### 2. **camel_b.obj** (211 KB) ✓ Verified
- **Intersections:** 23 pairs, 20 faces affected
- Full camel body mesh
- Moderate intersection count
- Best for: Testing on organic shapes

### 3. **cow.off** (182 KB) ✓ Verified
- **Intersections:** 83 pairs, 81 faces affected
- Classic cow mesh
- More complex intersections
- Best for: Medium complexity testing

### 4. **truck.obj** (163 KB) ✓ Verified
- **Intersections:** 424 pairs, 417 faces affected
- Truck/vehicle mesh with many intersections
- Most complex case in this set
- Best for: Stress testing repair algorithms

## Usage in ComfyUI

1. Load any mesh using the **Load Mesh** node
2. Connect to **Detect Self Intersections** node to visualize problems
3. Use **Remesh Self Intersections** node to attempt repairs
4. View results with **Preview Mesh VTK (Hidable Menu)** node

## Detection Requirements

The self-intersection detection and repair nodes require:
- `libigl` Python bindings (with CGAL support)
- Already installed in the `cad_node` conda environment

## Mesh Statistics

| Mesh | Intersection Pairs | Affected Faces | Size |
|------|-------------------|----------------|------|
| camelhead.off | 5 | 6 | 749 KB |
| camel_b.obj | 23 | 20 | 211 KB |
| cow.off | 83 | 81 | 182 KB |
| truck.obj | 424 | 417 | 163 KB |

## Verified Detection Results

When using **Detect Self Intersections** node, these are the confirmed results:
- **camelhead.off**: ✓ Detects 5 intersection pairs (6 faces affected)
- **camel_b.obj**: ✓ Detects 23 intersection pairs (20 faces affected)
- **cow.off**: ✓ Detects 83 intersection pairs (81 faces affected)
- **truck.obj**: ✓ Detects 424 intersection pairs (417 faces affected)

## Visualization

Use **Preview Mesh (VTK with Fields)** node to visualize:
- Scalar field `self_intersecting` on faces (1.0 = intersecting, 0.0 = valid)
- Scalar field `intersection_flag` on vertices (1.0 = adjacent to intersection)
- Scalar field `intersection_count` showing number of intersecting faces per vertex

## Repair Options

**Remesh Self Intersections** node parameters:
- `detect_only=False` - Actually remesh (subdivide intersecting triangles)
- `remove_unreferenced=True` - Clean up unused vertices
- `extract_outer_hull=False` - Extract outer hull for manifold result (slow but clean)
- `stitch_all=True` - Attempt to stitch boundaries

## Scripts

- **test_intersections.py** - Script to test meshes for self-intersections
- **intersection_test_results.txt** - Test results log

## Notes

- These are real-world meshes from graphics research, not synthetic test cases
- Complexity ranges from minimal (camelhead) to significant (truck)
- All meshes successfully tested with libigl CGAL detection
- Original source: [libigl tutorial data repository](https://github.com/alecjacobson/libigl-tutorial-data)

## Attribution

Test meshes sourced from libigl tutorial data (Alec Jacobson et al.)
