# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh UV Node - Remesh with texture preservation using Blender
"""

import numpy as np
import trimesh as trimesh_module
import os
import subprocess
import tempfile

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def _get_torch():
    """Lazy torch import to avoid importing before ComfyUI startup."""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _extract_texture(mesh):
    """Extract texture from trimesh visual to temp file."""
    if not PIL_AVAILABLE:
        return None, None

    if not hasattr(mesh, 'visual'):
        return None, None

    uvs = None
    if hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv

    texture_image = None
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material

        # Check for standard material.image (OBJ/MTL files)
        if hasattr(material, 'image'):
            img = material.image
            if isinstance(img, Image.Image):
                texture_image = img
                print(f"[TextureExtract] Found texture in material.image: {texture_image.size}")
            elif isinstance(img, str) and os.path.exists(img):
                texture_image = Image.open(img)
                print(f"[TextureExtract] Loaded texture from material.image path: {texture_image.size}")

        # Check for PBR baseColorTexture (GLB/GLTF files)
        if texture_image is None and hasattr(material, 'baseColorTexture'):
            img = material.baseColorTexture
            if isinstance(img, Image.Image):
                texture_image = img
                print(f"[TextureExtract] Found texture in material.baseColorTexture: {texture_image.size}")
            elif isinstance(img, str) and os.path.exists(img):
                texture_image = Image.open(img)
                print(f"[TextureExtract] Loaded texture from material.baseColorTexture path: {texture_image.size}")

        # Fallback: Check for main texture property
        if texture_image is None and hasattr(material, 'main'):
            img = material.main
            if isinstance(img, Image.Image):
                texture_image = img
                print(f"[TextureExtract] Found texture in material.main: {texture_image.size}")
            elif isinstance(img, str) and os.path.exists(img):
                texture_image = Image.open(img)
                print(f"[TextureExtract] Loaded texture from material.main path: {texture_image.size}")

    is_placeholder = False
    if texture_image is None and uvs is not None:
        # Create checkerboard placeholder
        print("[WARNING] Mesh has UVs but no texture image - using placeholder checkerboard")
        print("[WARNING] For proper texture baking, input mesh must have embedded texture data")
        is_placeholder = True
        texture_image = Image.new('RGB', (512, 512), color=(200, 200, 200))
        arr = np.array(texture_image)
        for i in range(0, 512, 64):
            for j in range(0, 512, 64):
                if (i // 64 + j // 64) % 2 == 0:
                    arr[i:i+64, j:j+64] = [100, 100, 100]
        texture_image = Image.fromarray(arr)

    if texture_image is None:
        return None, uvs, False

    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    texture_image.save(temp_file.name)
    temp_file.close()
    return temp_file.name, uvs, is_placeholder


def _load_as_comfy_image(texture_path):
    """Convert texture to ComfyUI IMAGE format."""
    torch = _get_torch()
    if not PIL_AVAILABLE or torch is None:
        return None

    img = Image.open(texture_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


class RemeshWithTexture:
    """
    Remesh with texture preservation using Blender baking.

    Workflow:
    1. Remeshes the input geometry using selected algorithm
    2. Generates new UV coordinates for the remeshed surface
    3. Bakes the original texture onto the new UVs using Blender
    4. Returns remeshed mesh with baked texture
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "method": (["blender", "xatlas"], {"default": "blender"}),
                "remesh_method": (["voxel", "quadriflow"], {"default": "quadriflow"}),
                "voxel_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100
                }),
                "bake_margin": ("INT", {
                    "default": 48,
                    "min": 0,
                    "max": 128,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "baked_texture", "info")
    FUNCTION = "remesh_with_texture"
    CATEGORY = "geompack/texture_remeshing"

    def remesh_with_texture(self, trimesh, method, remesh_method, voxel_size, target_face_count,
                           bake_margin):
        """Remesh a textured mesh while preserving texture through Python closest-point projection."""
        raise NotImplementedError(
            "Remesh with Texture is not yet implemented. "
            "Please use regular remeshing nodes instead."
        )

        if not PIL_AVAILABLE:
            raise RuntimeError("PIL required. Install: pip install Pillow")

        if _get_torch() is None:
            raise RuntimeError("torch required. Install: pip install torch")

        from ..._utils import blender_bridge
        from ..._utils import mesh_ops

        print(f"[BlenderRemeshWithTexture] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[BlenderRemeshWithTexture] Using Python texture transfer (no Blender baking)")

        # Extract texture from source mesh
        texture_path, original_uvs, is_placeholder = _extract_texture(trimesh)
        if texture_path is None:
            raise ValueError("Input mesh must have texture data (UVs and texture image)")

        # Auto-detect source texture size to preserve resolution
        if PIL_AVAILABLE:
            source_tex = Image.open(texture_path)
            source_tex_size = max(source_tex.size)  # Use max dimension if not square
            # Cap at 2048 to avoid memory issues with high poly meshes
            # (4096x4096 with 128 samples on complex meshes can crash Blender)
            actual_texture_size = min(source_tex_size, 2048)
            if actual_texture_size < source_tex_size:
                print(f"[BlenderRemeshWithTexture] Auto-detected texture size: {source_tex_size}x{source_tex_size}, capped at {actual_texture_size}x{actual_texture_size} to avoid memory issues")
            else:
                print(f"[BlenderRemeshWithTexture] Auto-detected texture size: {actual_texture_size}x{actual_texture_size}")
        else:
            actual_texture_size = 2048  # Fallback default
            print(f"[BlenderRemeshWithTexture] PIL not available, using default texture size: {actual_texture_size}x{actual_texture_size}")

        # Find Blender
        blender_path = blender_bridge.find_blender()

        # Create temp files (use GLB for both source and output to preserve materials)
        source_glb = tempfile.NamedTemporaryFile(suffix='_source.glb', delete=False)
        output_glb = tempfile.NamedTemporaryFile(suffix='_output.glb', delete=False)
        baked_texture = tempfile.NamedTemporaryFile(suffix='_baked.png', delete=False)
        baked_roughness = tempfile.NamedTemporaryFile(suffix='_roughness.png', delete=False)

        try:
            # Export source mesh as GLB (GLB format automatically includes textures)
            trimesh.export(source_glb.name)
            source_glb.close()
            output_glb.close()  # Close so Blender can write to it
            baked_texture.close()  # Close so Blender can write to it
            baked_roughness.close()  # Close so Blender can write to it

            # Calculate appropriate voxel size based on mesh bounds
            bounds = trimesh.bounds
            mesh_size = (bounds[1] - bounds[0]).max()
            # Adjust voxel size to be relative to mesh dimensions
            adjusted_voxel_size = voxel_size * mesh_size / 10.0

            # Calculate voxel size for fallback that targets approximate face count
            # Rough heuristic: voxel_size = mesh_size / (target_faces ** (1/3) * 2)
            fallback_voxel_size = mesh_size / (target_face_count ** (1/3) * 2)

            print(f"[BlenderRemeshWithTexture] Mesh bounds: {bounds}")
            print(f"[BlenderRemeshWithTexture] Mesh size: {mesh_size:.3f}")
            print(f"[BlenderRemeshWithTexture] Original voxel_size: {voxel_size}, Adjusted: {adjusted_voxel_size:.5f}")
            print(f"[BlenderRemeshWithTexture] Fallback voxel size (targeting ~{target_face_count} faces): {fallback_voxel_size:.5f}")

            # Build Blender script (remeshing will be applied to remeshed_obj)
            if remesh_method == "voxel":
                remesh_code = f"""
remeshed_obj.data.remesh_voxel_size = {adjusted_voxel_size}
original_face_count = len(remeshed_obj.data.polygons)
print(f"[Blender] Using voxel size: {adjusted_voxel_size}")
bpy.ops.object.voxel_remesh()
new_face_count = len(remeshed_obj.data.polygons)
print(f"[Blender] Voxel remesh: {{original_face_count}} -> {{new_face_count}} faces")
"""
            else:
                remesh_code = f"""
original_face_count = len(remeshed_obj.data.polygons)
print(f"[Blender] Starting quadriflow remesh with target_faces={target_face_count}")

# Try quadriflow, fall back to voxel if it fails
try:
    result = bpy.ops.object.quadriflow_remesh(target_faces={target_face_count})
    print(f"[Blender] Quadriflow result: {{result}}")
    new_face_count = len(remeshed_obj.data.polygons)

    # If face count unchanged, quadriflow silently failed - use voxel fallback
    if new_face_count == original_face_count:
        print("[Blender] WARNING: Quadriflow produced no change - falling back to voxel remesh")
        print(f"[Blender] Using fallback voxel size (targeting ~{target_face_count} faces): {fallback_voxel_size}")
        remeshed_obj.data.remesh_voxel_size = {fallback_voxel_size}
        bpy.ops.object.voxel_remesh()
        new_face_count = len(remeshed_obj.data.polygons)
        print(f"[Blender] Voxel remesh (fallback): {{original_face_count}} -> {{new_face_count}} faces")
    else:
        print(f"[Blender] Quadriflow remesh: {{original_face_count}} -> {{new_face_count}} faces")

except Exception as e:
    print(f"[Blender] ERROR: Quadriflow failed: {{e}}")
    print("[Blender] Falling back to voxel remesh")
    print(f"[Blender] Using fallback voxel size (targeting ~{target_face_count} faces): {fallback_voxel_size}")
    remeshed_obj.data.remesh_voxel_size = {fallback_voxel_size}
    bpy.ops.object.voxel_remesh()
    new_face_count = len(remeshed_obj.data.polygons)
    print(f"[Blender] Voxel remesh (fallback): {{original_face_count}} -> {{new_face_count}} faces")
"""

            script = f"""
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh with original UVs (GLB preserves materials better than OBJ)
bpy.ops.import_scene.gltf(filepath='{source_glb.name}')
objs = bpy.context.selected_objects

print(f"[Blender] Imported {{len(objs)}} objects")
for i, o in enumerate(objs):
    if o.type == 'MESH':
        print(f"  Object {{i}}: {{o.name}}, {{len(o.data.vertices)}} verts, {{len(o.data.polygons)}} faces, {{len(o.data.materials)}} materials")
    else:
        print(f"  Object {{i}}: {{o.name}} (type: {{o.type}}, skipping non-mesh)")

# Filter to only mesh objects
mesh_objs = [o for o in objs if o.type == 'MESH']
print(f"[Blender] Found {{len(mesh_objs)}} mesh objects out of {{len(objs)}} total objects")

if len(mesh_objs) == 0:
    raise RuntimeError("No mesh objects found in imported GLB file")

# If multiple mesh objects, join them
if len(mesh_objs) > 1:
    print(f"[Blender] Joining {{len(mesh_objs)}} mesh objects...")
    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]

    try:
        bpy.ops.object.join()
        obj = bpy.context.active_object
        print(f"[Blender] Join successful: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")
    except Exception as e:
        print(f"[Blender] ERROR joining objects: {{e}}")
        # If join fails, just use the first mesh object
        print(f"[Blender] WARNING: Using first mesh object only")
        obj = mesh_objs[0]
elif len(mesh_objs) == 1:
    obj = mesh_objs[0]

obj.name = 'OriginalMesh'

# Validate that we have a valid mesh object
if obj.type != 'MESH':
    raise RuntimeError(f"Final object is not a mesh (type: {{obj.type}})")
if not hasattr(obj.data, 'vertices') or not hasattr(obj.data, 'polygons'):
    raise RuntimeError("Final object does not have valid mesh data")

print(f"[Blender] Original mesh: {{len(obj.data.vertices)}} verts, {{len(obj.data.polygons)}} faces")

# Verify original mesh has UVs
if not obj.data.uv_layers.active:
    raise RuntimeError("Original mesh has no UV layer!")
print(f"[Blender] Original UV layer: {{obj.data.uv_layers.active.name}}")

# DUPLICATE the mesh - one for source (original UVs), one for target (remeshed)
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.duplicate()
remeshed_obj = bpy.context.active_object
remeshed_obj.name = 'RemeshedMesh'
print(f"[Blender] Created duplicate for remeshing")

# REMESH the duplicate (keeps original intact)
bpy.context.view_layer.objects.active = remeshed_obj
{remesh_code}

# Create NEW UV layer for remeshed geometry
new_uv = remeshed_obj.data.uv_layers.new(name="RemeshedUV")
remeshed_obj.data.uv_layers.active = new_uv

# Generate NEW UVs (high quality settings for better texture coverage)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(
    angle_limit=89.0,
    island_margin=0.001,
    scale_to_bounds=True,  # Scale UVs to fill [0,1] texture space
    correct_aspect=True    # Respect texture aspect ratio
)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"[Blender] Generated new UV layer 'RemeshedUV' on remeshed object")

# Export ONLY the remeshed object (no texture baking - Python will handle texture transfer)
bpy.ops.object.select_all(action='DESELECT')
remeshed_obj.select_set(True)
bpy.context.view_layer.objects.active = remeshed_obj

print(f"[Blender] Exporting remeshed object: {{len(remeshed_obj.data.vertices)}} vertices, {{len(remeshed_obj.data.polygons)}} faces")
bpy.ops.export_scene.gltf(
    filepath='{output_glb.name}',
    use_selection=True,
    export_format='GLB',
    export_texcoords=True,
    export_materials='EXPORT'
)
print(f"[Blender] Export complete")
"""

            print(f"[BlenderRemeshWithTexture] Running Blender...")
            # DEBUG: Save script for inspection
            with open('/tmp/blender_remesh_script.py', 'w') as f:
                f.write(script)
            print(f"[BlenderRemeshWithTexture] Script saved to /tmp/blender_remesh_script.py")

            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"[BlenderRemeshWithTexture] Blender stderr: {result.stderr}")
                print(f"[BlenderRemeshWithTexture] Blender stdout: {result.stdout}")
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Log Blender output for debugging
            if result.stdout:
                # Always show Blender's remesh feedback
                for line in result.stdout.split('\n'):
                    if '[Blender]' in line:
                        print(line)
                # Show ALL output if there's an error or warning
                if "error" in result.stdout.lower() or "warning" in result.stdout.lower() or "traceback" in result.stdout.lower():
                    print(f"[BlenderRemeshWithTexture] Full Blender output (error detected):")
                    print(result.stdout[-2000:])

            # Check if output file was created
            if not os.path.exists(output_glb.name) or os.path.getsize(output_glb.name) == 0:
                print(f"[BlenderRemeshWithTexture] ERROR: Output GLB not created or is empty!")
                print(f"[BlenderRemeshWithTexture] Full Blender output:")
                print(result.stdout[-2000:])
                raise RuntimeError("Blender did not create output GLB file - check logs above")

            # Load remeshed mesh
            print(f"[BlenderRemeshWithTexture] Loading GLB from: {output_glb.name}")
            remeshed = trimesh_module.load(output_glb.name, process=False)
            print(f"[BlenderRemeshWithTexture] Loaded type: {type(remeshed)}")

            if isinstance(remeshed, trimesh_module.Scene):
                print(f"[BlenderRemeshWithTexture] Scene contains {len(remeshed.geometry)} geometries")
                # Concatenate
                remeshed = remeshed.dump(concatenate=True)
                print(f"[BlenderRemeshWithTexture] After concatenate: {len(remeshed.vertices)} vertices, {len(remeshed.faces)} faces")

            # Merge duplicate vertices from GLB export
            remeshed.merge_vertices()
            print(f"[BlenderRemeshWithTexture] After merge_vertices: {len(remeshed.vertices)} vertices, {len(remeshed.faces)} faces")

            # Validate remesh results
            if len(remeshed.faces) == len(trimesh.faces):
                print(f"[WARNING] Face count unchanged ({len(remeshed.faces)}) - remesh may have failed!")
                print(f"[WARNING] Check Blender output above for details")

            # PYTHON TEXTURE TRANSFER: Use closest-point projection instead of Blender baking
            print(f"[BlenderRemeshWithTexture] Applying Python texture transfer...")
            remeshed_with_colors = mesh_ops.transfer_texture_via_closest_point(trimesh, remeshed)

            # Create visualization texture from vertex colors for IMAGE output
            # Simple approach: render a small texture showing the vertex color distribution
            if hasattr(remeshed_with_colors.visual, 'vertex_colors'):
                # Create a simple 256x256 texture showing vertex colors (for visualization)
                viz_size = 256
                vertex_colors_rgb = remeshed_with_colors.visual.vertex_colors[:, :3]  # RGB only

                # Reshape to image (simple grid layout)
                num_verts = len(vertex_colors_rgb)
                grid_size = int(np.ceil(np.sqrt(num_verts)))

                # Pad to fill grid
                padded_colors = np.zeros((grid_size * grid_size, 3), dtype=np.uint8)
                padded_colors[:num_verts] = vertex_colors_rgb

                # Reshape to 2D grid
                color_grid = padded_colors.reshape((grid_size, grid_size, 3))

                # Resize to viz_size using PIL
                color_img = Image.fromarray(color_grid)
                color_img = color_img.resize((viz_size, viz_size), Image.NEAREST)

                # Convert to ComfyUI format [1, H, W, 3] float32 in [0,1]
                comfy_image = np.array(color_img).astype(np.float32) / 255.0
                comfy_image = comfy_image[np.newaxis, ...]  # Add batch dimension

                # Convert to torch if needed
                torch = _get_torch()
                if torch is not None:
                    comfy_image = torch.from_numpy(comfy_image)

                print(f"[BlenderRemeshWithTexture] Created vertex color visualization: {viz_size}x{viz_size}")
            else:
                # Fallback: black image
                comfy_image = np.zeros((1, 256, 256, 3), dtype=np.float32)
                torch = _get_torch()
                if torch is not None:
                    comfy_image = torch.from_numpy(comfy_image)

            placeholder_warning = "\n⚠️  WARNING: Used placeholder texture (no embedded texture in input)" if is_placeholder else ""
            info = f"""Remesh with Texture (Python Transfer)
Method: {remesh_method}
Vertices: {len(trimesh.vertices)} -> {len(remeshed_with_colors.vertices)}
Faces: {len(trimesh.faces)} -> {len(remeshed_with_colors.faces)}
Texture Transfer: Closest-Point Projection{placeholder_warning}
"""

            print(f"[BlenderRemeshWithTexture] Complete")
            return (remeshed_with_colors, comfy_image, info)

        finally:
            # Cleanup
            for path in [source_glb.name, output_glb.name, texture_path, baked_texture.name, baked_roughness.name]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackRemeshWithTexture": RemeshWithTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemeshWithTexture": "Remesh with Texture",
}
