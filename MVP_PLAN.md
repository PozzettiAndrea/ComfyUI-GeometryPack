# ComfyUI libigl Wrapper - MVP Implementation Plan

## Project Overview

**Goal**: Create a comprehensive ComfyUI custom node package that wraps libigl geometry processing library for mesh manipulation workflows.

**Target Users**:
- General-purpose mesh processing
- High-poly mesh handling (>100K faces)
- Visualization essential in ComfyUI
- Standalone tool (no dependency on other 3D extensions)

**Core Technology Stack**:
- **libigl**: Python bindings for C++ geometry processing library
- **NumPy**: Array operations and mesh data storage
- **ComfyUI**: Node-based workflow system

---

## Project Structure

```
ComfyUI-libigl/
├── __init__.py              # Node registration and mappings
├── nodes.py                 # Node class implementations
├── mesh_utils.py            # Mesh data structures and utilities
├── requirements.txt         # Python dependencies
├── pyproject.toml          # ComfyUI registry metadata
├── README.md               # Documentation
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
├── examples/              # Test meshes
│   ├── cube.obj
│   └── bunny.obj
└── workflows/             # Example ComfyUI workflows
    └── basic_workflow.json
```

---

## Technical Architecture

### Mesh Data Type Definition

**Custom Type**: `"MESH"`

**Data Structure** (Python dict):
```python
mesh_data = {
    "vertices": np.ndarray,  # Shape: (N, 3) - XYZ coordinates
    "faces": np.ndarray,     # Shape: (F, 3) - Triangle vertex indices
    "normals": np.ndarray,   # Optional: (N, 3) - Per-vertex normals
    "vertex_colors": np.ndarray,  # Optional: (N, 3) or (N, 4) - RGB/RGBA colors
    "metadata": {
        "file_path": str,
        "is_manifold": bool,
        "has_boundary": bool,
        # ... other computed properties
    }
}
```

### Key Design Decisions

1. **NumPy-based**: Use NumPy arrays (not PyTorch tensors) for simplicity and libigl compatibility
2. **File Formats**: Support OBJ, PLY, STL (most common)
3. **Error Handling**: Validate mesh integrity, provide clear error messages
4. **Performance**: For high-poly meshes, implement progress indicators where possible
5. **Visualization**: Export mesh previews as images or point clouds for ComfyUI display

---

## MVP Scope: Phase 1 (Essential Nodes)

### Total: 14 Core Nodes

#### 1. I/O Nodes (3 nodes)
**Priority: CRITICAL**

1. **Load_Mesh**
   - Load mesh from file (OBJ/PLY/STL)
   - Inputs: file_path (STRING)
   - Outputs: MESH
   - Features: Auto-detect format, validate mesh

2. **Save_Mesh**
   - Save mesh to file
   - Inputs: MESH, file_path (STRING), format (COMBO)
   - Outputs: STRING (success message)
   - Features: Support OBJ, PLY, STL formats

3. **Mesh_Info**
   - Display mesh statistics
   - Inputs: MESH
   - Outputs: STRING (formatted stats)
   - Features: Vertex/face count, bounds, manifold status, boundary info

#### 2. Visualization Nodes (2 nodes)
**Priority: CRITICAL** (user requirement)

4. **Preview_Mesh**
   - Visualize mesh in ComfyUI
   - Inputs: MESH, camera_angle (COMBO), resolution (INT)
   - Outputs: IMAGE
   - Features: Render mesh to image using matplotlib or trimesh

5. **Mesh_To_PointCloud**
   - Convert mesh to point cloud for preview
   - Inputs: MESH, sample_count (INT)
   - Outputs: IMAGE
   - Features: Sample points on surface, render as scatter plot

#### 3. Primitive Generation (1 node)
**Priority: HIGH** (useful for testing)

6. **Create_Primitive**
   - Generate basic shapes
   - Inputs: shape (COMBO: cube/sphere/cylinder), size (FLOAT)
   - Outputs: MESH
   - Features: Generate test geometry

#### 4. Essential Cleanup (3 nodes)
**Priority: HIGH** (common operations)

7. **Remove_Duplicates**
   - Merge duplicate vertices/faces
   - Inputs: MESH, tolerance (FLOAT)
   - Outputs: MESH
   - Features: Configurable epsilon threshold

8. **Remove_Degenerate**
   - Delete zero-area faces
   - Inputs: MESH
   - Outputs: MESH
   - Features: Clean up bad geometry

9. **Compute_Normals**
   - Calculate vertex/face normals
   - Inputs: MESH, mode (COMBO: vertex/face)
   - Outputs: MESH (with normals)
   - Features: Essential for rendering and visualization

#### 5. Core Processing (3 nodes)
**Priority: HIGH** (key operations for high-poly)

10. **Decimate**
    - Reduce mesh complexity
    - Inputs: MESH, target_faces (INT or %), method (COMBO)
    - Outputs: MESH
    - Features: Critical for high-poly optimization

11. **Subdivide**
    - Increase mesh resolution
    - Inputs: MESH, iterations (INT), method (COMBO: loop/linear)
    - Outputs: MESH
    - Features: Loop subdivision, linear subdivision

12. **Smooth_Laplacian**
    - Smooth mesh surface
    - Inputs: MESH, iterations (INT), lambda (FLOAT)
    - Outputs: MESH
    - Features: Laplacian smoothing with iteration control

#### 6. Basic Utilities (2 nodes)
**Priority: MEDIUM**

13. **Transform_Mesh**
    - Apply transformations
    - Inputs: MESH, translate (VECTOR3), rotate (VECTOR3), scale (FLOAT/VECTOR3)
    - Outputs: MESH
    - Features: Basic transformations

14. **Bounding_Box**
    - Compute AABB
    - Inputs: MESH
    - Outputs: MESH (box mesh), STRING (bounds info)
    - Features: Get min/max bounds

---

## Phase 2: Extended Functionality (13 nodes)

**Priority: MEDIUM** (Implement after MVP is stable)

### Cleanup & Repair (3 nodes)
- **Fix_Normals**: Consistent face winding
- **Fill_Holes**: Close boundary loops
- **Split_Components**: Separate disconnected pieces

### Mesh Operations (4 nodes)
- **Merge_Meshes**: Combine multiple meshes
- **Slice_Plane**: Cut mesh with plane
- **Crop_Mesh**: Bounding box crop
- **Boolean_Operations**: Union/difference/intersection (if supported)

### Analysis (6 nodes)
- **Boundary_Loops**: Extract boundaries
- **Mesh_Volume**: Volume computation
- **Face_Areas**: Per-face area array
- **Field_To_Colors**: Scalar → vertex colors (for heatmap viz)
- **Store_Vertex_Field**: Custom per-vertex data
- **Get_Vertex_Field**: Extract field data

---

## Phase 3: Advanced Features (8 nodes)

**Priority: LOW** (Nice-to-have, implement if time permits)

### Deformation (2 nodes)
- **ARAP_Deform**: As-rigid-as-possible deformation
- **Harmonic_Deform**: Smooth deformation with constraints

### Distance & Queries (3 nodes)
- **Signed_Distance**: SDF at query points
- **Heat_Geodesic**: Fast geodesic distances
- **Closest_Point**: Nearest point on surface

### Curvature (1 node)
- **Principal_Curvature**: k1, k2 curvature + directions

### Sampling (2 nodes)
- **Sample_Points**: Uniform surface sampling
- **Poisson_Disk_Sample**: Blue noise sampling

---

## Never Implement (23 nodes)

**Reason: Redundant, too specialized, or better handled elsewhere**

### UV/Parameterization (3 nodes)
- **Harmonic_UV**: Basic unwrapping (Blender better)
- **LSCM_UV**: Conformal mapping (niche)
- **Transfer_UV**: UV copying (rare use case)

### Too Slow/Specialized (5 nodes)
- **Exact_Geodesic**: Very slow for high-poly
- **Intrinsic_Delaunay**: Complex remeshing
- **Ambient_Occlusion**: Better in rendering engines
- **Geodesic_Path**: Limited workflow use
- **Ray_Mesh_Intersect**: Low-level, better tools exist

### Redundant/Over-engineered (15 nodes)
- **Marching_Cubes**: Other tools handle SDF→mesh
- **Store_Face_Field**, **Get_Face_Field**: Over-engineered
- **Delete_Field**, **List_Fields**: Unnecessary complexity
- **Visualize_Field**: Covered by Field_To_Colors
- **Boolean_Preview**: Redundant with main preview
- **Compare_Meshes**: Niche analysis
- **Mesh_Statistics**: Covered by Mesh_Info
- **Vertex_Valence**: Too low-level
- **Face_Adjacency**: Too low-level
- **Edge_Topology**: Too low-level
- **Winding_Number**: Rarely needed in workflows
- **Point_Cloud_Distance**: Niche comparison
- **Mean_Curvature**, **Gaussian_Curvature**: Covered by Principal_Curvature
- **Transfer_Field**: Complex interpolation (niche)

---

## Implementation Roadmap

### Step 1: Project Setup (Day 1)
- [ ] Create directory structure
- [ ] Write `requirements.txt`
- [ ] Write `pyproject.toml`
- [ ] Write `.gitignore`
- [ ] Write `README.md` (basic)
- [ ] Add `LICENSE` (MIT)

### Step 2: Core Infrastructure (Day 1-2)
- [ ] Create `mesh_utils.py`:
  - Mesh data structure definition
  - Validation functions
  - File I/O wrappers (libigl)
  - Conversion utilities
- [ ] Test basic mesh loading/saving

### Step 3: Registration Boilerplate (Day 2)
- [ ] Create `__init__.py`:
  - NODE_CLASS_MAPPINGS
  - NODE_DISPLAY_NAME_MAPPINGS
- [ ] Create `nodes.py` skeleton

### Step 4: I/O Nodes (Day 2-3)
- [ ] Implement Load_Mesh
- [ ] Implement Save_Mesh
- [ ] Implement Mesh_Info
- [ ] Test with sample meshes

### Step 5: Visualization (Day 3-4)
- [ ] Implement Preview_Mesh (matplotlib/trimesh)
- [ ] Implement Mesh_To_PointCloud
- [ ] Test rendering pipeline

### Step 6: Core Processing (Day 4-6)
- [ ] Implement Create_Primitive
- [ ] Implement Decimate
- [ ] Implement Subdivide
- [ ] Implement Smooth_Laplacian
- [ ] Implement Compute_Normals
- [ ] Test with high-poly meshes

### Step 7: Cleanup & Utilities (Day 6-7)
- [ ] Implement Remove_Duplicates
- [ ] Implement Remove_Degenerate
- [ ] Implement Transform_Mesh
- [ ] Implement Bounding_Box

### Step 8: Testing & Polish (Day 7-8)
- [ ] Test all nodes with various meshes
- [ ] Error handling improvements
- [ ] Performance optimization
- [ ] Create example workflows
- [ ] Update README with examples

### Step 9: Documentation (Day 8)
- [ ] Write node documentation
- [ ] Create tutorial workflow
- [ ] Add screenshots to README

### Step 10: Release MVP (Day 8)
- [ ] Final testing
- [ ] Tag v0.1.0
- [ ] Publish to GitHub

---

## Dependencies

### `requirements.txt`
```
libigl>=2.6.1
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
Pillow>=9.0.0
```

### Optional (for enhanced visualization)
```
trimesh>=3.15.0
```

---

## Testing Strategy

### Test Meshes
1. **Simple**: cube.obj (8 vertices, 12 faces)
2. **Medium**: bunny.obj (~5K faces)
3. **High-poly**: dragon.obj (~100K faces)
4. **Edge cases**:
   - Non-manifold mesh
   - Mesh with boundaries
   - Degenerate triangles

### Test Workflows
1. **Basic Pipeline**: Load → Info → Preview
2. **Cleanup Pipeline**: Load → Remove Duplicates → Remove Degenerate → Preview
3. **Simplification**: Load → Decimate → Save
4. **Smoothing**: Load → Smooth → Preview
5. **Transformation**: Create Primitive → Transform → Preview

---

## Success Criteria for MVP

1. ✅ All 14 Phase 1 nodes implemented and working
2. ✅ Can load common mesh formats (OBJ, PLY, STL)
3. ✅ Can visualize meshes in ComfyUI
4. ✅ Can process high-poly meshes (>100K faces) without crashes
5. ✅ Decimation reduces mesh complexity effectively
6. ✅ All nodes have proper error handling
7. ✅ Example workflows demonstrate key features
8. ✅ README documentation is complete

---

## Future Enhancements (Post-MVP)

### Phase 2 Priority
1. Boolean operations (if libigl supports)
2. Field-based visualization (heatmaps)
3. Advanced cleanup (hole filling, component separation)

### Phase 3 Priority
1. Deformation tools (ARAP)
2. Geodesic computations
3. Advanced sampling

### Long-term
1. GPU acceleration for large meshes
2. Real-time preview updates
3. Custom shader support
4. Integration with ComfyUI-3D-Pack (if requested)

---

## Risk Mitigation

### Potential Issues

1. **High-poly performance**:
   - Solution: Progress indicators, chunked processing

2. **Visualization memory**:
   - Solution: Downsampling for preview, configurable resolution

3. **libigl limitations**:
   - Solution: Research libigl Python API capabilities early

4. **ComfyUI integration issues**:
   - Solution: Study existing 3D node packages, follow patterns

---

## Resources

- **libigl Python Tutorial**: https://libigl.github.io/libigl-python-bindings/
- **ComfyUI Custom Node Guide**: https://docs.comfy.org/essentials/custom_node_walkthrough
- **Template Repository**: https://github.com/jhj0517/ComfyUI-CustomNodes-Template
- **Example 3D Pack**: https://github.com/MrForExample/ComfyUI-3D-Pack

---

## Contact & Contribution

This is an MVP implementation plan. Feedback and contributions welcome!

**Estimated MVP Completion**: 8-10 days
**Total Nodes in MVP**: 14 nodes
**Total Nodes (all phases)**: ~35 nodes
