# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

#!/usr/bin/env python3
"""
Automated node splitting script for GeometryPack.
Splits monolithic node files into individual files per category.
"""

import re
import os
from pathlib import Path

# Define node extraction mappings
# Format: (source_file, category_dir, [(node_class, output_file, display_name)])
SPLITS = [
    # 1. repair/ - 9 nodes from repair.py
    ("repair.py", "repair", [
        ("FixNormalsNode", "fix_normals.py", "Fix Normals"),
        ("CheckNormalsNode", "check_normals.py", "Check Normals"),
        ("ComputeNormalsNode", "compute_normals.py", "Compute Normals"),
        ("VisualizNormalFieldNode", "visualize_normals.py", "Visualize Normal Field"),
        ("FillHolesNode", "fill_holes.py", "Fill Holes"),
        ("DetectSelfIntersectionsNode", "detect_intersections.py", "Detect Self Intersections"),
        ("RemeshSelfIntersectionsNode", "remesh_intersections.py", "Remesh Self Intersections"),
        ("FixSelfIntersectionsByRemovalNode", "fix_intersections_removal.py", "Fix Self Intersections By Removal"),
        ("FixSelfIntersectionsByPerturbationNode", "fix_intersections_perturbation.py", "Fix Self Intersections By Perturbation"),
    ]),

    # 2. uv/ - 5 ACTIVE nodes from uv.py
    ("uv.py", "uv", [
        ("XAtlasUVUnwrapNode", "xatlas_unwrap.py", "xAtlas UV Unwrap"),
        ("LibiglLSCMNode", "libigl_lscm.py", "libigl LSCM Unwrap"),
        ("LibiglHarmonicNode", "libigl_harmonic.py", "libigl Harmonic Unwrap"),
        ("LibiglARAPNode", "libigl_arap.py", "libigl ARAP Unwrap"),
        ("BlenderUVNode", "blender_uv.py", "Blender UV"),
    ]),

    # 3. visualization/ - 8 nodes from visualization.py
    ("visualization.py", "visualization", [
        ("PreviewMeshNode", "preview_mesh.py", "Preview Mesh (3D)"),
        ("PreviewMeshVTKNode", "preview_mesh_vtk.py", "Preview Mesh (VTK)"),
        ("PreviewMeshVTKHidableMenuNode", "preview_mesh_vtk_split.py", "Preview Mesh VTK (Hidable Menu)"),
        ("PreviewMeshVTKFiltersNode", "preview_mesh_vtk_with_texture.py", "Preview Mesh (VTK with Filters)"),
        ("PreviewMeshVTKFieldsNode", "preview_mesh_vtk_pointcloud.py", "Preview Mesh (VTK with Fields)"),
        ("PreviewMeshVTKTexturedNode", "preview_mesh_vtk_with_normals.py", "Preview Mesh (VTK with Textures)"),
        ("PreviewBoundingBoxesVTKNode", "preview_mesh_vtk_edges.py", "Preview Bounding Boxes (VTK)"),
        ("PreviewMeshUVNode", "preview_mesh_uv.py", "Preview Mesh (UV Layout)"),
    ]),

    # 4. transforms/ - 1 node
    ("transforms.py", "transforms", [
        ("TransformNode", "transform.py", "Transform"),
    ]),

    # 5. boolean/ - 1 node
    ("boolean.py", "boolean", [
        ("BooleanOpNode", "boolean_op.py", "Boolean"),
    ]),

    # 6. combine/ - 3 nodes from combine.py
    ("combine.py", "combine", [
        ("CombineMeshesNode", "combine_meshes.py", "Combine Meshes"),
        ("SplitComponentsNode", "combine_meshes_weighted.py", "Split Components"),
        ("FilterComponentsNode", "append_mesh.py", "Filter Components"),
    ]),

    # 7. distance/ - 2 nodes
    ("distance.py", "distance", [
        ("MeshDistanceNode", "mesh_to_mesh_distance.py", "Mesh Distance"),
        ("ComputeSDFNode", "point_to_mesh_distance.py", "Compute SDF"),
    ]),

    # 8. reconstruction/ - 1 node
    ("reconstruction.py", "reconstruction", [
        ("ReconstructSurfaceNode", "reconstruct_surface.py", "Reconstruct Surface"),
    ]),

    # 9. skeleton/ - 2 nodes
    ("skeleton.py", "skeleton", [
        ("ExtractSkeleton", "extract_skeleton.py", "Extract Skeleton"),
        ("SkeletonToTrimesh", "mesh_from_skeleton.py", "Skeleton to Lines"),
    ]),

    # 10. texture_remeshing/ - 2 nodes
    ("texture_remeshing.py", "texture_remeshing", [
        ("BlenderRemeshWithTexture", "remesh_uv.py", "Blender Remesh with Texture"),
        ("XAtlasRemeshWithTexture", "texture_to_geometry.py", "xAtlas Remesh with Texture"),
    ]),

    # 11. conversion/ - 2 nodes
    ("conversion.py", "conversion", [
        ("StripMeshAdjacencyNode", "mesh_to_pointcloud.py", "Strip Mesh Adjacency"),
        ("MeshToPointCloudNode", "pointcloud_to_mesh.py", "Mesh to Point Cloud"),
    ]),

    # 12. analysis/ - 2 nodes
    ("analysis.py", "analysis", [
        ("MeshInfoNode", "mesh_info.py", "Mesh Info"),
        ("MarkBoundaryEdgesNode", "mesh_quality.py", "Mark Boundary Edges"),
    ]),

    # 13. examples/ - 1 node
    ("examples.py", "examples", [
        ("ExampleLibiglNode", "example.py", "Example Node"),
    ]),
]


def extract_class_code(content, class_name):
    """Extract a class and its helper functions from source content."""
    # Find the class definition
    class_pattern = rf'^class {class_name}.*?(?=^class |^NODE_CLASS_MAPPINGS|^# Node mappings|$)'
    match = re.search(class_pattern, content, re.MULTILINE | re.DOTALL)

    if not match:
        return None

    return match.group(0).rstrip()


def extract_imports(content):
    """Extract all import statements from source content."""
    import_pattern = r'^(?:import|from)\s+.+$'
    imports = re.findall(import_pattern, content, re.MULTILINE)

    # Update import paths
    updated_imports = []
    for imp in imports:
        # Replace relative imports
        if 'from . import' in imp:
            imp = imp.replace('from . import', 'from .. import')
        updated_imports.append(imp)

    return '\n'.join(updated_imports)


def extract_helper_functions(content, class_name):
    """Extract helper functions used by a class."""
    # Find functions defined before the class that might be helpers
    before_class = content.split(f'class {class_name}')[0]

    # Extract function definitions (simple heuristic)
    func_pattern = r'^def [_a-z][a-z0-9_]*\([^)]*\):[^\n]*\n(?:    .+\n)*'
    functions = re.findall(func_pattern, before_class, re.MULTILINE)

    return '\n\n'.join(functions) if functions else ""


def create_node_file(source_content, class_name, output_path, display_name, category_name):
    """Create an individual node file."""
    # Extract imports
    imports = extract_imports(source_content)

    # Extract helper functions
    helpers = extract_helper_functions(source_content, class_name)

    # Extract class code
    class_code = extract_class_code(source_content, class_name)

    if not class_code:
        print(f"  [WARN] Could not extract {class_name}")
        return False

    # Determine node key from class name
    node_key = f"GeomPack{class_name.replace('Node', '').replace('class', '')}"

    # Build output file content
    output_content = f'''"""
{category_name.title()} Nodes - {display_name}
"""

{imports}

{helpers}

{class_code}


# Node mappings
NODE_CLASS_MAPPINGS = {{
    "{node_key}": {class_name},
}}

NODE_DISPLAY_NAME_MAPPINGS = {{
    "{node_key}": "{display_name}",
}}
'''

    # Write file
    with open(output_path, 'w') as f:
        f.write(output_content)

    print(f"  [OK] Created {output_path.name}")
    return True


def create_init_file(category_dir, nodes):
    """Create __init__.py for a category that aggregates all nodes."""
    imports = []
    class_mappings = []
    display_mappings = []

    for class_name, filename, display_name in nodes:
        module_name = filename.replace('.py', '')
        imports.append(f"from .{module_name} import NODE_CLASS_MAPPINGS as {module_name.upper()}_MAPPINGS")
        imports.append(f"from .{module_name} import NODE_DISPLAY_NAME_MAPPINGS as {module_name.upper()}_DISPLAY")
        class_mappings.append(f"    **{module_name.upper()}_MAPPINGS,")
        display_mappings.append(f"    **{module_name.upper()}_DISPLAY,")

    init_content = f'''"""
{category_dir.name.title()} category node aggregation
"""

{chr(10).join(imports)}

# Aggregate all node mappings
NODE_CLASS_MAPPINGS = {{
{chr(10).join(class_mappings)}
}}

NODE_DISPLAY_NAME_MAPPINGS = {{
{chr(10).join(display_mappings)}
}}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
'''

    init_path = category_dir / "__init__.py"
    with open(init_path, 'w') as f:
        f.write(init_content)

    print(f"  [OK] Created {category_dir.name}/__init__.py")


def main():
    """Main execution."""
    nodes_dir = Path(__file__).parent

    print("=" * 60)
    print("GeometryPack Node Splitting Automation")
    print("=" * 60)

    for source_file, category_name, nodes in SPLITS:
        print(f"\n[DIR] Processing {category_name}/ ({len(nodes)} nodes)")

        # Read source file
        source_path = nodes_dir / source_file
        if not source_path.exists():
            print(f"  [WARN] Source file not found: {source_file}")
            continue

        with open(source_path, 'r') as f:
            source_content = f.read()

        # Create category directory
        category_dir = nodes_dir / category_name
        category_dir.mkdir(exist_ok=True)

        # Extract each node
        for class_name, output_filename, display_name in nodes:
            output_path = category_dir / output_filename
            create_node_file(source_content, class_name, output_path, display_name, category_name)

        # Create __init__.py
        create_init_file(category_dir, nodes)

    print("\n" + "=" * 60)
    print("[OK] Splitting complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
