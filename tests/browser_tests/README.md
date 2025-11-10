# Browser Tests for ComfyUI GeometryPack

Browser automation tests for VTK viewer UI using Selenium and pytest.

## Overview

These tests verify the VTK mesh viewer functionality in a real browser environment, including:
- File format detection (STL, OBJ, VTP)
- Viewer initialization and loading
- Scalar field visualization
- Widget-viewer integration

## Requirements

- **Selenium 4.15+** (already installed in your environment)
- **ComfyUI running** on `http://localhost:8188` (or custom URL)
- **Chrome/Safari/Firefox** browser installed

## Quick Start

### 1. Start ComfyUI

```bash
# Make sure ComfyUI is running
python main.py --port 8188
```

### 2. Run All Browser Tests

```bash
# Run all browser tests
pytest tests/browser_tests/ -v

# Run with visible browser (not headless)
pytest tests/browser_tests/ -v --headed

# Run specific test file
pytest tests/browser_tests/test_vtk_viewer.py -v

# Run specific test
pytest tests/browser_tests/test_vtk_viewer.py::TestVTKViewer::test_vtp_format_supported -v
```

### 3. View Results

- **Screenshots on failure**: Saved to `test-screenshots/`
- **Console logs on failure**: Saved to `test-screenshots/*.log`

## Test Files

### `test_vtk_viewer.py`
Tests for the basic VTK viewer (`viewer_vtk.html`):
- ✅ VTP format support (the bug we fixed)
- ✅ STL/OBJ format still work
- ✅ File format detection with query strings
- ✅ Viewer controls present
- ✅ Unsupported formats rejected

### `test_vtk_filters.py`
Tests for the advanced filters viewer (`viewer_vtk_filters.html`):
- ✅ VTP file loading
- ✅ Scalar field selector exists
- ✅ VTP prioritized over STL
- ✅ postMessage communication
- ✅ Multiple format support

### `test_vtk_fields.py`
Tests for PreviewMeshVTKFields widget integration:
- ✅ Widget loads correct viewer (filters, not basic)
- ✅ Scalar field selector present
- ✅ VTP files accepted
- ✅ postMessage integration
- ✅ Comparison between viewers

## Command Line Options

```bash
# Browser selection
pytest tests/browser_tests/ --browser=chrome   # Default
pytest tests/browser_tests/ --browser=safari
pytest tests/browser_tests/ --browser=firefox

# Headed mode (see the browser)
pytest tests/browser_tests/ --headed

# Custom ComfyUI URL
pytest tests/browser_tests/ --comfyui-url=http://localhost:8000

# Run only browser tests
pytest -m browser -v

# Skip browser tests
pytest -m "not browser" -v
```

## Writing New Tests

### Basic Test Template

```python
import pytest

@pytest.mark.browser
class TestMyFeature:
    def test_something(self, viewer_page):
        # Load the viewer
        viewer_page.load_viewer("viewer_vtk.html")

        # Send a mesh load message
        viewer_page.send_load_mesh_message("test.vtp")

        # Check console logs
        logs = viewer_page.get_console_logs()
        assert "VTP" in "\n".join(logs)

        # Check elements
        assert viewer_page.check_element_exists("#myElement")
```

## Fixtures

### `browser`
Selenium WebDriver instance (Chrome/Safari/Firefox)
- Auto-captures screenshots on failure
- Auto-captures console logs on failure
- Configurable headless/headed mode

### `viewer_page`
Helper class for VTK viewer interactions:
- `load_viewer(name)` - Load a specific viewer HTML
- `send_load_mesh_message(filename)` - Simulate ComfyUI postMessage
- `get_console_logs()` - Get browser console output
- `check_element_exists(selector)` - Check if element present
- `wait_for_text(text)` - Wait for text to appear

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
- name: Install Chrome
  uses: browser-actions/setup-chrome@latest

- name: Start ComfyUI
  run: |
    python main.py --port 8188 &
    sleep 10  # Wait for server to start

- name: Run browser tests
  run: |
    pytest tests/browser_tests/ -v --headed=false

- name: Upload screenshots on failure
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: test-screenshots
    path: test-screenshots/
```

## Troubleshooting

### Browser driver not found
```bash
# Install webdriver-manager for automatic driver management
pip install webdriver-manager

# Or manually download ChromeDriver
# https://chromedriver.chromium.org/downloads
```

### ComfyUI not running
```bash
# Make sure ComfyUI is started first
python main.py --port 8188

# Or use custom URL
pytest tests/browser_tests/ --comfyui-url=http://your-server:8188
```

### Tests failing with timeout
- Increase timeout in `conftest.py`
- Check if ComfyUI is accessible
- Run with `--headed` to see what's happening

### Can't see what's happening
```bash
# Run with visible browser
pytest tests/browser_tests/ --headed -v

# Add more sleeps to slow down
# Edit test files and increase time.sleep() values
```

## What These Tests Verify

These browser tests were created to verify the fix for:

**Issue**: PreviewMeshVTKFields node couldn't display VTP files or scalar fields

**Root Cause**:
1. `viewer_vtk.html` didn't support VTP format
2. `mesh_preview_vtk_fields.js` loaded wrong viewer (basic instead of filters)

**Fix Applied**:
1. Added VTP support to `viewer_vtk.html` (imported `vtkXMLPolyDataReader`)
2. Changed `mesh_preview_vtk_fields.js` to load `viewer_vtk_filters.html`

**Tests Verify**:
- ✅ VTP files now load in basic viewer
- ✅ Filters viewer has scalar field selector
- ✅ Fields widget uses correct viewer
- ✅ No regression in STL/OBJ support

## Examples

### Check if VTP fix works
```bash
pytest tests/browser_tests/test_vtk_viewer.py::TestVTKViewer::test_vtp_format_supported -v --headed
```

### Verify filters viewer has scalar selector
```bash
pytest tests/browser_tests/test_vtk_filters.py::TestVTKFiltersViewer::test_scalar_field_selector_exists -v --headed
```

### Compare basic vs filters viewer
```bash
pytest tests/browser_tests/test_vtk_fields.py::TestVTKFieldsWidget::test_comparison_basic_vs_filters_viewer -v --headed
```

## Notes

- Tests use real browser (Chrome by default)
- Tests can run headless or with visible browser
- Screenshots saved automatically on failure
- Console logs captured for debugging
- Works with existing pytest infrastructure
- Integrates with pytest markers (`-m browser`)

## Related Files

- `web/viewer_vtk.html` - Basic VTK viewer
- `web/viewer_vtk_filters.html` - Advanced viewer with scalar fields
- `web/js/mesh_preview_vtk_fields.js` - Fields widget (loads filters viewer)
- `web/js/mesh_preview_vtk_filters.js` - Filters widget
