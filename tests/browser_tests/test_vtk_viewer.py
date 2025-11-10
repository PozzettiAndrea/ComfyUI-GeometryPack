"""
Browser tests for basic VTK viewer (viewer_vtk.html).

Tests file format detection, VTP support, and basic mesh loading.
"""

import pytest
import time


@pytest.mark.browser
class TestVTKViewer:
    """Test suite for viewer_vtk.html - basic VTK viewer."""

    def test_viewer_loads(self, viewer_page):
        """Test that the basic VTK viewer HTML page loads successfully."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Check that the page loaded
        assert "GeomPack" in viewer_page.driver.title or \
               viewer_page.check_element_exists("#container"), \
               "VTK viewer page did not load correctly"

        # Check for loading indicator
        assert viewer_page.check_element_exists("#loading"), \
               "Loading indicator not found"

        print("✅ Viewer loaded successfully")

    def test_vtp_format_supported(self, viewer_page):
        """Test that VTP file format is now supported (the bug we fixed)."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Send a VTP file load message
        viewer_page.send_load_mesh_message("test_mesh.vtp")

        # Wait a bit for processing
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Check that VTP is detected
        assert "VTP" in log_text or ".vtp" in log_text, \
               "VTP file format not detected in logs"

        # Check that there's NO "Unsupported file format" error
        assert "Unsupported file format" not in log_text, \
               "VTP file format still showing as unsupported!"

        # Check that VTP reader is mentioned (if loaded successfully)
        if "VTP (XML PolyData)" in log_text or "VTP" in log_text:
            print("✅ VTP format is recognized")
        else:
            print("⚠️  VTP format detection unclear from logs")

        print(f"\n📝 Console logs:\n{log_text}")

    def test_stl_format_still_works(self, viewer_page):
        """Test that STL files still work after our VTP changes."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Send an STL file load message
        viewer_page.send_load_mesh_message("test_mesh.stl")
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Check that STL is detected
        assert "STL" in log_text or ".stl" in log_text, \
               "STL file format not detected"

        # Check that there's no unsupported format error
        assert "Unsupported file format" not in log_text, \
               "STL files broken after VTP changes"

        print("✅ STL format still works")

    def test_obj_format_still_works(self, viewer_page):
        """Test that OBJ files still work after our VTP changes."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Send an OBJ file load message
        viewer_page.send_load_mesh_message("test_mesh.obj")
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Check that OBJ is detected
        assert "OBJ" in log_text or ".obj" in log_text, \
               "OBJ file format not detected"

        # Check that there's no unsupported format error
        assert "Unsupported file format" not in log_text, \
               "OBJ files broken after VTP changes"

        print("✅ OBJ format still works")

    def test_unsupported_format_error(self, viewer_page):
        """Test that truly unsupported formats still show error."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Send an unsupported file format
        viewer_page.send_load_mesh_message("test_mesh.fbx")
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # Should show unsupported format error for FBX
        assert "Unsupported file format" in log_text or "supports STL, OBJ, and VTP" in log_text, \
               "Unsupported format (FBX) did not trigger error message"

        print("✅ Unsupported formats correctly rejected")

    def test_file_detection_in_query_string(self, viewer_page):
        """
        Test that file format detection works with query string URLs.

        This was the original bug - the URL has query parameters like:
        /view?filename=mesh.vtp&type=output&subfolder=
        """
        viewer_page.load_viewer("viewer_vtk.html")

        # Send a realistic ComfyUI-style URL with query params
        script = """
            window.postMessage({
                type: 'LOAD_MESH',
                filepath: '/view?filename=preview_vtk_fields_abc123.vtp&type=output&subfolder=',
                timestamp: Date.now()
            }, '*');
        """
        viewer_page.driver.execute_script(script)
        time.sleep(2)

        # Get console logs
        logs = viewer_page.get_console_logs()
        log_text = "\n".join(logs)

        # The .vtp extension should be detected even with query params
        assert ".vtp" in log_text, \
               "VTP extension not detected in query string URL"

        # Should not show unsupported format error
        assert "Unsupported file format" not in log_text, \
               "VTP file in query string URL not recognized"

        print("✅ File format detection works with query string URLs")

    def test_controls_present(self, viewer_page):
        """Test that viewer controls are present."""
        viewer_page.load_viewer("viewer_vtk.html")

        # Check for control elements
        assert viewer_page.check_element_exists("#controls"), \
               "Controls panel not found"

        assert viewer_page.check_element_exists("#showEdges"), \
               "Show edges checkbox not found"

        assert viewer_page.check_element_exists("#representation"), \
               "Representation selector not found"

        assert viewer_page.check_element_exists("#resetCamera"), \
               "Reset camera button not found"

        assert viewer_page.check_element_exists("#screenshot"), \
               "Screenshot button not found"

        print("✅ All viewer controls present")
