# ComfyUI-GeometryPack

Professional geometry processing nodes for ComfyUI. Load, analyze, remesh, unwrap, and visualize 3D meshes directly in your workflows.

**[View Workflow Gallery](https://pozzettiandrea.github.io/ComfyUI-GeometryPack/)**

## Some previews

![Remeshing](docs/remeshing.png)
![self_intersection_fix](docs/self_intersection_fix.png)
![uv_unwrapping](docs/uv_unwrapping.png)

https://github.com/user-attachments/assets/c2e393ba-fb57-4a46-b772-31c4b01f47f8


https://github.com/user-attachments/assets/b1bcd570-719f-4be5-868d-f72c9f1f4c10


https://github.com/user-attachments/assets/607e6b29-a8d4-4346-873a-5bd6b140bdba


https://github.com/user-attachments/assets/7718a5d7-cd5b-47d3-874c-29ada4320694



## Features

**Core Operations**
- Load/Save meshes (OBJ, PLY, STL, OFF, etc.)
- Primitive generation (cube, sphere, plane)
- Mesh analysis and statistics
- Interactive 3D preview (Three.js & VTK.js)

**Remeshing**
- PyMeshLab isotropic remeshing
- CGAL isotropic remeshing (optional)
- Blender voxel & quadriflow remeshing

**UV Mapping**
- xAtlas UV unwrapping (fast, no dependencies)
- libigl LSCM conformal mapping
- Blender projections (cube, cylinder, sphere)

**Analysis**
- Boundary edge detection
- Hausdorff & Chamfer distance
- Signed distance fields (SDF)
- Point cloud conversion

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/PozzettiAndrea/ComfyUI-GeometryPack.git
cd ComfyUI-GeometryPack
pip install -r requirements.txt
```

**Automatic Blender Installation (Recommended):**
```bash
python install.py
```

This will automatically download and install a portable version of Blender for UV unwrapping and remeshing nodes. No admin rights required!

Restart ComfyUI. Nodes appear in the `geompack/` category.

**Optional Dependencies:**
- **Blender**: Auto-installed via `install.py`, or install manually from [blender.org](https://www.blender.org/download/)
- **CGAL**: Build tools for CGAL remeshing (see `cgal_tools/README.md`)

## Quick Start

**Basic workflow:**
```
Create Primitive → Mesh Info → Preview Mesh (3D)
Load Mesh → PyMeshLab Remesh → Save Mesh
```

## Workflow Demonstrations

### Loading & Analysis

**Load Mesh (Blend/FBX)**
![Load Blend or FBX](docs/workflows/load_blend_or_fbx.png)

**Mesh Information**
![Mesh Info](docs/workflows/mesh_info.png)

### Remeshing & Refinement

**Remeshing**
![Remesh](docs/workflows/remesh.png)

**Batch Remeshing**
![Batch Remesh](docs/workflows/batch_remesh.png)

**Mesh Refinement**
![Refine](docs/workflows/refine.png)

### Mesh Repair

**Fill Holes**
![Fill Holes](docs/workflows/fill_holes.png)

**Self-Intersection Removal**
![Self Intersection Removal](docs/workflows/self_intersection_removal.png)

**Self-Intersection Remesh**
![Self Intersection Remesh](docs/workflows/self_intersection_remesh.png)

### Boolean Operations

**Boolean Operations**
![Boolean Ops](docs/workflows/boolean_ops.png)

### Distance & Analysis

**Hausdorff & Chamfer Distance**
![Hausdorff or Chamfer Distance](docs/workflows/hausdorff_or_chamfer_distance.png)

**Signed Distance Function**
![Signed Distance Function](docs/workflows/signed_distance_function.png)

### Advanced Features

**Skeleton Extraction**
![Skeleton Extraction](docs/workflows/skeleton_extraction.png)

**Normal Fields Visualization**
![Normal Fields](docs/workflows/normal_fields.png)

**Transform Operations**
![Transform Operations](docs/workflows/transform_operations.png)

### Texture & Conversion

**Preview with Texture**
![Preview with Texture](docs/workflows/preview_with_texture.png)

**Depth Map to Mesh**
![Depth Map to Mesh](docs/workflows/depth_map_to_mesh.png)

**Alpha Wrapping**
![uv_unwrapping](docs/uv_unwrapping.png)

## Demo Videos

*Coming soon - video demonstrations will be added here*

## Architecture

Codebase organized by function in `nodes/` directory:
- **io.py** - Load/Save
- **primitives.py** - Shape generation
- **analysis.py** - Mesh info, boundary detection
- **distance.py** - Hausdorff, Chamfer, SDF
- **conversion.py** - Mesh to point cloud
- **remeshing.py** - PyMeshLab, CGAL, Blender
- **uv.py** - UV unwrapping (xAtlas, libigl, Blender)
- **transforms.py** - Positioning
- **visualization.py** - 3D preview (Three.js, VTK.js)

All nodes use `trimesh.Trimesh` objects for mesh data.

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

Built on [trimesh](https://trimesh.org/), [libigl](https://libigl.github.io/), [PyMeshLab](https://pymeshlab.readthedocs.io/), and [CGAL](https://www.cgal.org/).

## License

**GNU General Public License v3.0 or later (GPL-3.0-or-later)**

This project is licensed under the GPL-3.0-or-later license to ensure compatibility with the included dependencies:

- **Blender** (GPL-2.0-or-later) - Used for advanced UV unwrapping and remeshing
- **CGAL** (GPL-3.0-or-later) - Used for boolean operations and isotropic remeshing
- **PyMeshLab** (GPL-3.0) - Used for mesh processing operations

### What This Means

- ✅ You can use, modify, and distribute this software freely
- ✅ You can use it for commercial purposes
- ⚠️ If you distribute modified versions, you must also license them under GPL-3.0-or-later
- ⚠️ You must share the source code of any modifications you distribute

For more details, see:
- [LICENSE](LICENSE) - Full GPL-3.0 license text
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) - Detailed third-party license information

### Questions?

If you have questions about licensing, please open an issue on [GitHub](https://github.com/PozzettiAndrea/ComfyUI-GeometryPack/issues).
