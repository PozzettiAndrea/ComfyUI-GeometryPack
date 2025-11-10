"""Pytest configuration and fixtures for ComfyUI-GeometryPack tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np
import trimesh

# Add parent directory to path so we can import nodes
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock ComfyUI modules before importing nodes
sys.modules["folder_paths"] = MagicMock()
sys.modules["folder_paths"].get_input_directory = MagicMock(
    return_value=str(Path(__file__).parent / "assets")
)
sys.modules["folder_paths"].get_output_directory = MagicMock(
    return_value=str(Path(__file__).parent / "outputs" / "meshes")
)

# Set up paths
TEST_DIR = Path(__file__).parent.parent  # Go up to tests/ directory
ASSETS_DIR = TEST_DIR / "assets"
OUTPUT_DIR = TEST_DIR / "outputs"
MESHES_DIR = OUTPUT_DIR / "meshes"
RENDERS_DIR = OUTPUT_DIR / "renders"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no optional dependencies)")
    config.addinivalue_line("markers", "optional: Tests requiring optional dependencies")
    config.addinivalue_line("markers", "blender: Tests requiring Blender")
    config.addinivalue_line("markers", "slow: Slow tests (heavy processing)")


@pytest.fixture(scope="session", autouse=True)
def setup_output_dirs():
    """Create output directories for test artifacts."""
    MESHES_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def test_assets_dir():
    """Return path to test assets directory."""
    return ASSETS_DIR


@pytest.fixture
def test_output_dir(request):
    """Return path to test-specific output directory."""
    # Create a directory for this specific test
    test_name = request.node.name
    test_dir = OUTPUT_DIR / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def meshes_output_dir(test_output_dir):
    """Return path to test-specific meshes directory."""
    mesh_dir = test_output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    return mesh_dir


@pytest.fixture
def renders_output_dir(test_output_dir):
    """Return path to test-specific renders directory."""
    render_dir = test_output_dir / "renders"
    render_dir.mkdir(exist_ok=True)
    return render_dir


@pytest.fixture
def bunny_mesh():
    """Load Stanford Bunny mesh."""
    bunny_path = ASSETS_DIR / "Stanford_Bunny.stl"
    if bunny_path.exists():
        return trimesh.load(str(bunny_path))
    return None


@pytest.fixture
def cube_mesh():
    """Create a simple cube mesh."""
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def sphere_mesh():
    """Create a simple sphere mesh."""
    return trimesh.creation.icosphere(subdivisions=2, radius=1.0)


@pytest.fixture
def open_mesh():
    """Create an open mesh (cylinder without caps) for UV unwrapping tests."""
    # Create a cylinder without endcaps (open mesh)
    cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=16)
    return cylinder


def render_mesh_to_png(mesh, output_path, resolution=(800, 800)):
    """Render mesh to PNG file for visual verification."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh collection
        collection = Poly3DCollection(
            mesh.vertices[mesh.faces],
            alpha=0.7,
            facecolor='cyan',
            edgecolor='black',
            linewidths=0.1
        )
        ax.add_collection3d(collection)

        # Set axis limits
        scale = mesh.vertices.flatten()
        ax.auto_scale_xyz(scale, scale, scale)

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Warning: Could not render mesh: {e}")
        return False


@pytest.fixture
def render_helper(renders_output_dir):
    """Helper fixture for rendering meshes."""
    def _render(mesh, name, **kwargs):
        output_path = renders_output_dir / f"{name}.png"
        render_mesh_to_png(mesh, output_path, **kwargs)
        return output_path
    return _render


@pytest.fixture
def save_mesh_helper(meshes_output_dir):
    """Helper fixture for saving meshes."""
    def _save(mesh, name, format="obj"):
        output_path = meshes_output_dir / f"{name}.{format}"
        mesh.export(str(output_path))
        return output_path
    return _save


@pytest.fixture
def save_and_render(save_mesh_helper, render_helper):
    """Combined helper to save mesh and render it."""
    def _save_and_render(mesh, name, format="obj"):
        save_mesh_helper(mesh, name, format)
        render_helper(mesh, name)
        return True
    return _save_and_render
