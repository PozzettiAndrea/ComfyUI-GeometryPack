# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
ComfyUI GeometryPack Nodes
Organized by dependency environment: main, cgal, blender, gpu
"""

from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ==============================================================================
# Main nodes (always load - basic dependencies like trimesh, numpy, pymeshlab)
# ==============================================================================
from .main import NODE_CLASS_MAPPINGS as main_mappings
from .main import NODE_DISPLAY_NAME_MAPPINGS as main_display
NODE_CLASS_MAPPINGS.update(main_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(main_display)
print(f"[GeomPack] Main nodes loaded ({len(main_mappings)} nodes)")

# ==============================================================================
# Isolated nodes - wrapped for subprocess execution
# ==============================================================================
try:
    from comfy_env import wrap_isolated_nodes

    nodes_dir = Path(__file__).parent

    # CGAL nodes (isolated - C++ deps could conflict)
    try:
        from .cgal import NODE_CLASS_MAPPINGS as cgal_mappings
        from .cgal import NODE_DISPLAY_NAME_MAPPINGS as cgal_display
        cgal_wrapped = wrap_isolated_nodes(cgal_mappings, nodes_dir / "cgal")
        NODE_CLASS_MAPPINGS.update(cgal_wrapped)
        NODE_DISPLAY_NAME_MAPPINGS.update(cgal_display)
        print(f"[GeomPack] CGAL nodes loaded ({len(cgal_mappings)} nodes, isolated)")
    except ImportError as e:
        print(f"[GeomPack] CGAL nodes not available: {e}")

    # Blender nodes (isolated - needs Python 3.11)
    try:
        from .blender import NODE_CLASS_MAPPINGS as blender_mappings
        from .blender import NODE_DISPLAY_NAME_MAPPINGS as blender_display
        blender_wrapped = wrap_isolated_nodes(blender_mappings, nodes_dir / "blender")
        NODE_CLASS_MAPPINGS.update(blender_wrapped)
        NODE_DISPLAY_NAME_MAPPINGS.update(blender_display)
        print(f"[GeomPack] Blender nodes loaded ({len(blender_mappings)} nodes, isolated)")
    except ImportError as e:
        print(f"[GeomPack] Blender nodes not available: {e}")

    # GPU nodes (isolated - CUDA packages need specific PyTorch)
    try:
        from .gpu import NODE_CLASS_MAPPINGS as gpu_mappings
        from .gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display
        gpu_wrapped = wrap_isolated_nodes(gpu_mappings, nodes_dir / "gpu")
        NODE_CLASS_MAPPINGS.update(gpu_wrapped)
        NODE_DISPLAY_NAME_MAPPINGS.update(gpu_display)
        print(f"[GeomPack] GPU nodes loaded ({len(gpu_mappings)} nodes, isolated)")
    except ImportError as e:
        print(f"[GeomPack] GPU nodes not available: {e}")

except ImportError:
    print("[GeomPack] comfy-env not installed, isolated nodes disabled")
    print("[GeomPack] Install with: pip install comfy-env")

print(f"[GeomPack] Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
