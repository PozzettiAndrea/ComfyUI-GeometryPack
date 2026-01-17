# Third-Party Notices

This file contains notices for third-party software included with or used by ComfyUI-GeometryPack.

## License Summary

ComfyUI-GeometryPack is licensed under GPL-3.0-or-later. This project incorporates and depends on the following third-party software:

---

## Bundled Software

### Blender 4.2.3
- **License:** GNU General Public License v2.0 or later (GPL-2.0-or-later)
- **Copyright:** © 2023 Blender Foundation
- **Website:** https://www.blender.org
- **Location:** `_blender/blender-4.2.3-linux-x64/`
- **Size:** ~1.3GB
- **License File:** `_blender/blender-4.2.3-linux-x64/GPL-license.txt`
- **Third-Party Licenses:** `_blender/blender-4.2.3-linux-x64/license/THIRD-PARTY-LICENSES.txt`

Blender is used for advanced UV unwrapping and remeshing operations.

---

## Python Dependencies

### Core Geometry Processing

#### libigl
- **License:** Mozilla Public License 2.0 (MPL-2.0)
- **Website:** https://libigl.github.io/
- **Purpose:** Computational geometry library

#### trimesh
- **License:** MIT License
- **Website:** https://trimsh.org/
- **Purpose:** Mesh loading and processing

#### PyMeshLab
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Website:** https://github.com/cnr-isti-vclab/PyMeshLab
- **Purpose:** MeshLab integration for mesh processing

#### CGAL (Python bindings)
- **License:** GNU General Public License v3.0 or later (GPL-3.0-or-later) OR Commercial License
- **Website:** https://www.cgal.org/
- **Purpose:** Boolean operations, intersection detection, isotropic remeshing
- **Note:** GeometryPack uses the GPL-3.0 version

### UV and Point Cloud

#### xatlas
- **License:** MIT License
- **Website:** https://github.com/jpcy/xatlas
- **Purpose:** UV unwrapping

#### point-cloud-utils
- **License:** MIT License
- **Website:** https://github.com/fwilliams/point-cloud-utils
- **Purpose:** Point cloud processing and mesh sampling

#### fast-simplification
- **License:** MIT License
- **Website:** https://github.com/pyvista/fast-simplification
- **Purpose:** Mesh simplification

### Scientific Computing

#### NumPy
- **License:** BSD 3-Clause License
- **Website:** https://numpy.org/
- **Purpose:** Numerical computing

#### SciPy
- **License:** BSD 3-Clause License
- **Website:** https://scipy.org/
- **Purpose:** Scientific computing

### Visualization and Utilities

#### matplotlib
- **License:** Matplotlib License (PSF-based)
- **Website:** https://matplotlib.org/
- **Purpose:** Plotting and visualization

#### Pillow
- **License:** HPND License
- **Website:** https://python-pillow.org/
- **Purpose:** Image processing

#### requests
- **License:** Apache License 2.0
- **Website:** https://requests.readthedocs.io/
- **Purpose:** HTTP library

#### tqdm
- **License:** Mozilla Public License 2.0 (MPL-2.0) and MIT License
- **Website:** https://github.com/tqdm/tqdm
- **Purpose:** Progress bars

### Optional Dependencies

#### skeletor
- **License:** GNU General Public License v3.0 (GPL-3.0)
- **Website:** https://github.com/navis-org/skeletor
- **Purpose:** Skeleton extraction (optional)

#### rtree
- **License:** MIT License
- **Website:** https://github.com/Toblerity/rtree
- **Purpose:** Spatial indexing

---

## JavaScript Dependencies

### VTK.js
- **License:** BSD 3-Clause License
- **Website:** https://kitware.github.io/vtk-js/
- **Purpose:** 3D visualization in web browser
- **Location:** `web/js/vtk-gltf.js`

### Build Tools (Development Only)

#### webpack
- **License:** MIT License
- **Website:** https://webpack.js.org/
- **Purpose:** Module bundler (build-time only)

#### Babel
- **License:** MIT License
- **Website:** https://babeljs.io/
- **Purpose:** JavaScript transpiler (build-time only)

---

## License Compatibility Notice

This project is distributed under **GPL-3.0-or-later** to ensure compatibility with all included GPL-licensed components:
- Blender (GPL-2.0-or-later)
- CGAL (GPL-3.0-or-later)
- PyMeshLab (GPL-3.0)
- skeletor (GPL-3.0)

All other dependencies (MIT, BSD, Apache 2.0, MPL-2.0, etc.) are compatible with GPL-3.0.

---

## Full License Texts

The full text of the GPL-3.0 license can be found in the `LICENSE` file at the root of this repository.

For full license texts of third-party dependencies, please refer to:
- Blender licenses: `_blender/blender-4.2.3-linux-x64/license/`
- Python package licenses: Check individual package documentation or use `pip show <package-name>`
- JavaScript package licenses: See `build_vtk_bundle/node_modules/<package>/LICENSE`

---

## Attribution

ComfyUI-GeometryPack is built on the shoulders of giants. We are grateful to all the developers and contributors of the above projects for their excellent work.

For questions about licensing, please open an issue at:
https://github.com/PozzettiAndrea/ComfyUI-GeometryPack/issues
