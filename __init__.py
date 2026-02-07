# SPDX-License-Identifier: GPL-3.0-or-later
"""ComfyUI GeometryPack - Geometry Processing Custom Nodes."""

import os
import sys
from pathlib import Path

print("[geompack] loading...", file=sys.stderr, flush=True)
from comfy_env import register_nodes
print("[geompack] calling register_nodes", file=sys.stderr, flush=True)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


def _generate_widget_mappings():
    """Generate widget visibility mappings from node metadata."""
    try:
        from comfy_dynamic_widgets import scan_specific_nodes, generate_mappings
        import json

        configs = scan_specific_nodes(NODE_CLASS_MAPPINGS)
        if not configs:
            return

        mappings = generate_mappings(configs)
        output_path = os.path.join(os.path.dirname(__file__), "web", "js", "mappings.json")
        with open(output_path, "w") as f:
            json.dump(mappings, f, indent=2)
    except ImportError:
        pass
    except Exception:
        pass


_generate_widget_mappings()

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
