# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GPU-dependent nodes - requires CUDA, torch, and cumesh.
These nodes will only load if GPU dependencies are available.
"""

# Check for GPU availability at import time
def _check_gpu_available():
    """Check if GPU dependencies (CUDA, torch, cumesh) are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False

    try:
        import cumesh
        return True
    except ImportError:
        return False


if not _check_gpu_available():
    raise ImportError(
        "GPU dependencies not available. "
        "Requires CUDA-capable GPU, torch with CUDA support, and cumesh package."
    )

# Import submodules
from . import remeshing

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(remeshing.NODE_CLASS_MAPPINGS)

# Collect all display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(remeshing.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
