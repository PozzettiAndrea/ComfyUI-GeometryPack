import sys
print("[geompack] loading...", file=sys.stderr, flush=True)
from comfy_env import register_nodes
print("[geompack] calling register_nodes", file=sys.stderr, flush=True)

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()


from comfy_dynamic_widgets import write_mappings
write_mappings(NODE_CLASS_MAPPINGS, __file__)

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
