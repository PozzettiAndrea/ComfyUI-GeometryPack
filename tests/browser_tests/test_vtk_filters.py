"""
Browser tests for VTK filters viewer (viewer_vtk_filters.html).

Tests VTP loading, scalar field visualization, and color mapping.
"""

import pytest
import time


@pytest.mark.browser
class TestVTKFiltersViewer:
    """Test suite for viewer_vtk_filters.html - advanced viewer with scalar fields."""

    def test_filters_viewer_loads(self, viewer_page):
        """Test that the filters viewer HTML page loads successfully."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Check that the page loaded
        assert viewer_page.check_element_exists("#container"), \
               "VTK filters viewer page did not load correctly"

        # Check for loading indicator
        assert viewer_page.check_element_exists("#loading"), \
               "Loading indicator not found"

        print("✅ Filters viewer loaded successfully")

    def test_vtp_format_supported_filters(self, viewer_page):
        """Test that VTP files load in the filters viewer."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Send a VTP file load message
        viewer_page.send_load_mesh_message("test_with_fields.vtp")
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Check that VTP is detected (should be checked FIRST in filters viewer)
        assert "VTP" in log_text or ".vtp" in log_text, \
               "VTP file format not detected in filters viewer"

        # Check for specific filters viewer log message
        if "GeomPack VTK Filters" in log_text:
            print("✅ Filters viewer is responding")

        # Check that there's NO "Unsupported file format" error
        assert "Unsupported file format" not in log_text, \
               "VTP file format showing as unsupported in filters viewer!"

        print(f"\n📝 Console logs:\n{log_text}")

    def test_scalar_field_selector_exists(self, viewer_page):
        """Test that the scalar field selector control exists."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Check for scalar field selector
        has_selector = viewer_page.check_element_exists("#scalarFieldSelector")

        assert has_selector, \
               "Scalar field selector (#scalarFieldSelector) not found in filters viewer"

        print("✅ Scalar field selector found")

    def test_filter_controls_present(self, viewer_page):
        """Test that all filter controls are present."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Check for basic controls
        assert viewer_page.check_element_exists("#controls"), \
               "Controls panel not found"

        # Check for filter-specific controls
        assert viewer_page.check_element_exists("#scalarFieldSelector"), \
               "Scalar field selector not found"

        # Check for standard controls
        assert viewer_page.check_element_exists("#showEdges"), \
               "Show edges checkbox not found"

        assert viewer_page.check_element_exists("#representation"), \
               "Representation selector not found"

        assert viewer_page.check_element_exists("#resetCamera"), \
               "Reset camera button not found"

        print("✅ All filter controls present")

    def test_vtp_prioritized_over_stl(self, viewer_page):
        """
        Test that VTP detection happens BEFORE STL detection.

        In the filters viewer, VTP should be checked first since it preserves
        scalar field data.
        """
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Send a VTP file
        viewer_page.send_load_mesh_message("mesh_with_fields.vtp")
        time.sleep(2)

        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # In filters viewer, should see VTP detection
        # Look for the order of log messages
        vtp_index = log_text.find("VTP")
        stl_index = log_text.find("STL")

        if vtp_index > 0 and stl_index > 0:
            # Both found - VTP should come first
            assert vtp_index < stl_index, \
                   "VTP not prioritized over STL in filters viewer"
            print("✅ VTP correctly prioritized")
        elif vtp_index > 0:
            print("✅ VTP detected (STL not mentioned)")
        else:
            print("⚠️  Could not verify VTP prioritization from logs")

    def test_console_logs_use_filters_prefix(self, viewer_page):
        """Test that console logs use '[GeomPack VTK Filters]' prefix."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Trigger some activity
        viewer_page.send_load_mesh_message("test.vtp")
        time.sleep(2)

        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Should see the filters-specific log prefix
        has_filters_logs = "GeomPack VTK Filters" in log_text or \
                          "[GeomPack VTK Filters]" in log_text

        assert has_filters_logs, \
               "Filters viewer not using correct log prefix. Are we loading the wrong viewer?"

        print("✅ Correct log prefix found")

    def test_postmessage_received(self, viewer_page):
        """Test that the filters viewer receives postMessage events."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Send a message
        viewer_page.send_load_mesh_message("test_fields.vtp")
        time.sleep(2)

        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Check for postMessage receipt
        assert "Received postMessage" in log_text or \
               "LOAD_MESH" in log_text, \
               "Filters viewer not receiving postMessage events"

        print("✅ postMessage communication working")

    def test_multiple_file_formats(self, viewer_page):
        """Test that filters viewer supports multiple formats."""
        viewer_page.load_viewer("viewer_vtk_filters.html")

        # Test VTP
        viewer_page.send_load_mesh_message("mesh1.vtp")
        time.sleep(1)

        logs1 = viewer_page.get_console_logs()
        assert any("VTP" in log or ".vtp" in log for log in logs1), \
               "VTP not detected"

        # Reload viewer for fresh state
        viewer_page.load_viewer("viewer_vtk_filters.html")
        time.sleep(1)

        # Test STL
        viewer_page.send_load_mesh_message("mesh2.stl")
        time.sleep(1)

        logs2 = viewer_page.get_console_logs()
        assert any("STL" in log or ".stl" in log for log in logs2), \
               "STL not detected"

        # Reload viewer for fresh state
        viewer_page.load_viewer("viewer_vtk_filters.html")
        time.sleep(1)

        # Test OBJ
        viewer_page.send_load_mesh_message("mesh3.obj")
        time.sleep(1)

        logs3 = viewer_page.get_console_logs()
        assert any("OBJ" in log or ".obj" in log for log in logs3), \
               "OBJ not detected"

        print("✅ All formats detected in filters viewer")
