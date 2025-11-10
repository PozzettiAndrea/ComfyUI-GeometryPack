"""
Browser test fixtures for ComfyUI GeometryPack VTK viewers.

This module provides pytest fixtures for browser automation testing using Selenium.
"""

import pytest
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.fixture(scope="session")
def comfyui_url():
    """Base URL for ComfyUI server. Override with --comfyui-url option."""
    return os.environ.get("COMFYUI_URL", "http://localhost:8188")


@pytest.fixture(scope="function")
def browser(request, comfyui_url):
    """
    Create a Selenium WebDriver instance for browser testing.

    Supports Chrome, Safari, and Firefox. Chrome is the default.
    Use --headed flag to run with visible browser (default is headless).
    """
    # Get options from command line
    headed = request.config.getoption("--headed", default=False)
    browser_name = request.config.getoption("--browser", default="chrome").lower()

    # Set up Chrome (most common)
    if browser_name == "chrome":
        chrome_options = Options()
        if not headed:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        # Enable browser logging
        chrome_options.set_capability("goog:loggingPrefs", {"browser": "ALL"})

        driver = webdriver.Chrome(options=chrome_options)

    elif browser_name == "safari":
        # Safari on Mac
        driver = webdriver.Safari()
        driver.set_window_size(1920, 1080)

    elif browser_name == "firefox":
        firefox_options = webdriver.FirefoxOptions()
        if not headed:
            firefox_options.add_argument("--headless")
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")
        driver = webdriver.Firefox(options=firefox_options)

    else:
        raise ValueError(f"Unsupported browser: {browser_name}")

    # Set implicit wait
    driver.implicitly_wait(10)

    # Store base URL
    driver.comfyui_url = comfyui_url

    yield driver

    # Teardown: capture screenshot on failure
    if request.node.rep_call.failed:
        screenshot_dir = "test-screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(
            screenshot_dir,
            f"{request.node.name}_{int(time.time())}.png"
        )
        driver.save_screenshot(screenshot_path)
        print(f"\n📸 Screenshot saved: {screenshot_path}")

        # Capture browser console logs
        try:
            logs = driver.get_log("browser")
            log_path = os.path.join(
                screenshot_dir,
                f"{request.node.name}_{int(time.time())}_console.log"
            )
            with open(log_path, "w") as f:
                for log in logs:
                    f.write(f"[{log['level']}] {log['message']}\n")
            print(f"📝 Console logs saved: {log_path}")
        except Exception as e:
            print(f"⚠️  Could not capture console logs: {e}")

    driver.quit()


@pytest.fixture
def viewer_page(browser, comfyui_url):
    """
    Load a VTK viewer page and return helper methods for interacting with it.
    """
    class ViewerPage:
        def __init__(self, driver, base_url):
            self.driver = driver
            self.base_url = base_url

        def load_viewer(self, viewer_name="viewer_vtk.html"):
            """Load a specific VTK viewer HTML file."""
            url = f"{self.base_url}/extensions/ComfyUI-GeometryPack/{viewer_name}"
            print(f"🌐 Loading viewer: {url}")
            self.driver.get(url)
            time.sleep(2)  # Wait for vtk.js to initialize

        def send_load_mesh_message(self, filename, file_type="output", subfolder=""):
            """Send a LOAD_MESH postMessage to the viewer (simulating ComfyUI)."""
            filepath = f"/view?filename={filename}&type={file_type}&subfolder={subfolder}"
            script = f"""
                window.postMessage({{
                    type: 'LOAD_MESH',
                    filepath: '{filepath}',
                    timestamp: Date.now()
                }}, '*');
            """
            print(f"📤 Sending LOAD_MESH message: {filepath}")
            self.driver.execute_script(script)
            time.sleep(2)  # Wait for mesh to load

        def get_console_logs(self):
            """Get browser console logs."""
            try:
                logs = self.driver.get_log("browser")
                return [f"[{log['level']}] {log['message']}" for log in logs]
            except Exception as e:
                print(f"⚠️  Could not get console logs: {e}")
                return []

        def wait_for_text(self, text, timeout=10):
            """Wait for specific text to appear on the page."""
            WebDriverWait(self.driver, timeout).until(
                EC.text_to_be_present_in_element((By.TAG_NAME, "body"), text)
            )

        def check_element_exists(self, selector, by=By.CSS_SELECTOR):
            """Check if an element exists on the page."""
            try:
                self.driver.find_element(by, selector)
                return True
            except:
                return False

        def get_element_text(self, selector, by=By.CSS_SELECTOR):
            """Get text content of an element."""
            try:
                element = self.driver.find_element(by, selector)
                return element.text
            except:
                return None

    return ViewerPage(browser, comfyui_url)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to store test result in the request fixture for screenshot capture.
    """
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--headed",
        action="store_true",
        default=False,
        help="Run browser tests with visible browser window (not headless)"
    )
    parser.addoption(
        "--browser",
        action="store",
        default="chrome",
        help="Browser to use for tests (chrome, safari, firefox)"
    )
    parser.addoption(
        "--comfyui-url",
        action="store",
        default="http://localhost:8188",
        help="Base URL for ComfyUI server"
    )
