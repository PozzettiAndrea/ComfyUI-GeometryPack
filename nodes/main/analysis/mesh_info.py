# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh Info Node - Display detailed mesh information
"""

import numpy as np
import trimesh


def _extract_visual_info(mesh: trimesh.Trimesh) -> dict:
    info = {
        'visual_type': 'none',
        'has_material': False,
        'material_type': None,
        'has_uv': False,
        'uv_count': 0,
        'uv_range_u': None,
        'uv_range_v': None,
        'has_vertex_colors': False,
        'has_face_colors': False,
        'texture_dimensions': None,
        'texture_format': None,
    }

    if not hasattr(mesh, 'visual') or mesh.visual is None:
        return info

    visual = mesh.visual

    visual_class = type(visual).__name__
    if 'TextureVisuals' in visual_class:
        info['visual_type'] = 'texture'
    elif 'ColorVisuals' in visual_class:
        info['visual_type'] = 'color'
    else:
        info['visual_type'] = visual_class.lower()

    if hasattr(visual, 'material') and visual.material is not None:
        info['has_material'] = True
        info['material_type'] = type(visual.material).__name__

    if hasattr(visual, 'uv') and visual.uv is not None and len(visual.uv) > 0:
        info['has_uv'] = True
        info['uv_count'] = len(visual.uv)
        uv_array = np.array(visual.uv)
        info['uv_range_u'] = (float(uv_array[:, 0].min()), float(uv_array[:, 0].max()))
        info['uv_range_v'] = (float(uv_array[:, 1].min()), float(uv_array[:, 1].max()))

    if hasattr(visual, 'vertex_colors') and visual.vertex_colors is not None and len(visual.vertex_colors) > 0:
        info['has_vertex_colors'] = True

    if hasattr(visual, 'face_colors') and visual.face_colors is not None and len(visual.face_colors) > 0:
        info['has_face_colors'] = True

    if hasattr(visual, 'material') and visual.material is not None:
        material = visual.material
        texture_image = None
        if hasattr(material, 'image') and material.image is not None:
            texture_image = material.image
        elif hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_image = material.baseColorTexture

        if texture_image is not None:
            if hasattr(texture_image, 'size'):
                info['texture_dimensions'] = texture_image.size
                info['texture_format'] = texture_image.format or 'unknown'

    return info


def _extract_pbr_properties(material) -> dict:
    props = {
        'has_base_color_texture': False,
        'has_metallic_roughness_texture': False,
        'has_normal_texture': False,
        'has_occlusion_texture': False,
        'has_emissive_texture': False,
        'metallic_factor': None,
        'roughness_factor': None,
        'base_color_factor': None,
        'emissive_factor': None,
        'alpha_mode': None,
        'alpha_cutoff': None,
        'double_sided': None,
    }

    if material is None:
        return props

    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        props['has_base_color_texture'] = True
    if hasattr(material, 'metallicRoughnessTexture') and material.metallicRoughnessTexture is not None:
        props['has_metallic_roughness_texture'] = True
    if hasattr(material, 'normalTexture') and material.normalTexture is not None:
        props['has_normal_texture'] = True
    if hasattr(material, 'occlusionTexture') and material.occlusionTexture is not None:
        props['has_occlusion_texture'] = True
    if hasattr(material, 'emissiveTexture') and material.emissiveTexture is not None:
        props['has_emissive_texture'] = True
    if hasattr(material, 'metallicFactor'):
        props['metallic_factor'] = material.metallicFactor
    if hasattr(material, 'roughnessFactor'):
        props['roughness_factor'] = material.roughnessFactor
    if hasattr(material, 'baseColorFactor'):
        props['base_color_factor'] = material.baseColorFactor
    if hasattr(material, 'emissiveFactor'):
        props['emissive_factor'] = material.emissiveFactor
    if hasattr(material, 'alphaMode'):
        props['alpha_mode'] = material.alphaMode
    if hasattr(material, 'alphaCutoff'):
        props['alpha_cutoff'] = material.alphaCutoff
    if hasattr(material, 'doubleSided'):
        props['double_sided'] = material.doubleSided

    return props


def _extract_custom_attributes(mesh: trimesh.Trimesh) -> dict:
    attrs = {
        'vertex_attributes': {},
        'face_attributes': {},
    }

    if hasattr(mesh, 'vertex_attributes') and mesh.vertex_attributes:
        for name, values in mesh.vertex_attributes.items():
            attr_info = {
                'count': len(values),
                'dtype': str(values.dtype) if hasattr(values, 'dtype') else 'unknown',
                'shape': values.shape if hasattr(values, 'shape') else None,
            }
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.number):
                attr_info['min'] = float(np.min(values))
                attr_info['max'] = float(np.max(values))
            attrs['vertex_attributes'][name] = attr_info

    if hasattr(mesh, 'face_attributes') and mesh.face_attributes:
        for name, values in mesh.face_attributes.items():
            attr_info = {
                'count': len(values),
                'dtype': str(values.dtype) if hasattr(values, 'dtype') else 'unknown',
                'shape': values.shape if hasattr(values, 'shape') else None,
            }
            if hasattr(values, 'dtype') and np.issubdtype(values.dtype, np.number):
                attr_info['min'] = float(np.min(values))
                attr_info['max'] = float(np.max(values))
            attrs['face_attributes'][name] = attr_info

    return attrs


def _compute_mesh_info(mesh: trimesh.Trimesh) -> str:
    if not isinstance(mesh, trimesh.Trimesh):
        return "Error: Input must be a trimesh.Trimesh object"

    info = "=== Mesh Information ===\n\n"
    info += f"Vertices: {len(mesh.vertices):,}\n"
    info += f"Faces: {len(mesh.faces):,}\n"
    info += f"Edges: {len(mesh.edges):,}\n\n"

    info += f"Volume: {mesh.volume:.6f}\n"
    info += f"Surface Area: {mesh.area:.6f}\n"
    info += f"Is Watertight: {mesh.is_watertight}\n"
    info += f"Is Winding Consistent: {mesh.is_winding_consistent}\n\n"

    bounds = mesh.bounds
    center = mesh.centroid
    extents = mesh.extents

    info += "Bounding Box:\n"
    info += f"  Min: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}]\n"
    info += f"  Max: [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f}]\n"
    info += f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]\n"
    info += f"  Extents: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]\n\n"

    visual_info = _extract_visual_info(mesh)
    info += "=== Visual & Material ===\n\n"
    info += f"Visual Type: {visual_info['visual_type']}\n"
    info += f"Has Material: {visual_info['has_material']}\n"
    if visual_info['material_type']:
        info += f"Material Type: {visual_info['material_type']}\n"
    info += f"UV Coordinates: {'Yes' if visual_info['has_uv'] else 'No'}"
    if visual_info['has_uv']:
        info += f" ({visual_info['uv_count']:,} entries)\n"
        if visual_info['uv_range_u'] and visual_info['uv_range_v']:
            info += f"  UV Range: U[{visual_info['uv_range_u'][0]:.3f}, {visual_info['uv_range_u'][1]:.3f}], "
            info += f"V[{visual_info['uv_range_v'][0]:.3f}, {visual_info['uv_range_v'][1]:.3f}]\n"
    else:
        info += "\n"
    info += f"Vertex Colors: {'Yes' if visual_info['has_vertex_colors'] else 'No'}\n"
    info += f"Face Colors: {'Yes' if visual_info['has_face_colors'] else 'No'}\n"
    if visual_info['texture_dimensions']:
        info += f"Texture Dimensions: {visual_info['texture_dimensions'][0]}x{visual_info['texture_dimensions'][1]}\n"
        if visual_info['texture_format']:
            info += f"Texture Format: {visual_info['texture_format']}\n"
    info += "\n"

    if visual_info['has_material'] and hasattr(mesh.visual, 'material'):
        pbr_props = _extract_pbr_properties(mesh.visual.material)
        info += "=== PBR Material Properties ===\n\n"
        info += f"Base Color Texture: {'Yes' if pbr_props['has_base_color_texture'] else 'No'}\n"
        info += f"Metallic/Roughness Texture: {'Yes' if pbr_props['has_metallic_roughness_texture'] else 'No'}\n"
        info += f"Normal Map: {'Yes' if pbr_props['has_normal_texture'] else 'No'}\n"
        info += f"Occlusion Texture: {'Yes' if pbr_props['has_occlusion_texture'] else 'No'}\n"
        info += f"Emissive Texture: {'Yes' if pbr_props['has_emissive_texture'] else 'No'}\n"

        if pbr_props['metallic_factor'] is not None:
            info += f"Metallic Factor: {pbr_props['metallic_factor']:.3f}\n"
        if pbr_props['roughness_factor'] is not None:
            info += f"Roughness Factor: {pbr_props['roughness_factor']:.3f}\n"
        if pbr_props['base_color_factor'] is not None:
            bcf = pbr_props['base_color_factor']
            if hasattr(bcf, '__len__') and len(bcf) >= 3:
                info += f"Base Color Factor: [{bcf[0]:.3f}, {bcf[1]:.3f}, {bcf[2]:.3f}"
                if len(bcf) >= 4:
                    info += f", {bcf[3]:.3f}]\n"
                else:
                    info += "]\n"
            else:
                info += f"Base Color Factor: {bcf}\n"
        if pbr_props['emissive_factor'] is not None:
            ef = pbr_props['emissive_factor']
            if hasattr(ef, '__len__') and len(ef) >= 3:
                info += f"Emissive Factor: [{ef[0]:.3f}, {ef[1]:.3f}, {ef[2]:.3f}]\n"
            else:
                info += f"Emissive Factor: {ef}\n"
        if pbr_props['alpha_mode'] is not None:
            info += f"Alpha Mode: {pbr_props['alpha_mode']}\n"
        if pbr_props['alpha_cutoff'] is not None:
            info += f"Alpha Cutoff: {pbr_props['alpha_cutoff']:.3f}\n"
        if pbr_props['double_sided'] is not None:
            info += f"Double Sided: {pbr_props['double_sided']}\n"
        info += "\n"

    custom_attrs = _extract_custom_attributes(mesh)
    info += "=== Custom Attributes ===\n\n"

    if custom_attrs['vertex_attributes']:
        info += "Vertex Attributes:\n"
        for name, attr in custom_attrs['vertex_attributes'].items():
            info += f"  {name}: {attr['dtype']}"
            if attr['shape']:
                info += f" {attr['shape']}"
            if 'min' in attr and 'max' in attr:
                info += f" range=[{attr['min']:.3f}, {attr['max']:.3f}]"
            info += "\n"
    else:
        info += "Vertex Attributes: (none)\n"

    if custom_attrs['face_attributes']:
        info += "Face Attributes:\n"
        for name, attr in custom_attrs['face_attributes'].items():
            info += f"  {name}: {attr['dtype']}"
            if attr['shape']:
                info += f" {attr['shape']}"
            if 'min' in attr and 'max' in attr:
                info += f" range=[{attr['min']:.3f}, {attr['max']:.3f}]"
            info += "\n"
    else:
        info += "Face Attributes: (none)\n"

    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        info += f"\nVertex Normals: Yes ({len(mesh.vertex_normals):,} vectors)\n"

    if mesh.metadata:
        info += "\n=== Metadata ===\n\n"
        for key, value in mesh.metadata.items():
            info += f"  {key}: {value}\n"

    return info


class MeshInfoNode:
    """
    Display detailed mesh information and statistics.
    Now using trimesh for enhanced mesh analysis.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_NODE = True
    FUNCTION = "get_mesh_info"
    CATEGORY = "geompack/analysis"
    INPUT_IS_LIST = True

    def get_mesh_info(self, trimesh):
        """
        Get information about the trimesh(es).

        Args:
            trimesh: list of trimesh.Trimesh objects (when INPUT_IS_LIST=True)

        Returns:
            dict: {"result": (info_string,), "ui": {"text": [info]}}
        """
        # Handle batch processing - concatenate all info
        all_info = []
        for i, mesh in enumerate(trimesh):
            mesh_info = _compute_mesh_info(mesh)
            batch_header = f"{'='*60}\n=== Batch Item {i+1}/{len(trimesh)} ===\n{'='*60}\n\n"
            all_info.append(batch_header + mesh_info)

        # Join all info with separators
        combined_info = "\n\n".join(all_info)

        return {
            "result": (combined_info,),
            "ui": {"text": [combined_info]}
        }


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshInfo": MeshInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshInfo": "Mesh Info",
}
