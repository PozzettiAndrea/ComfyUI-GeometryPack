/**
 * SettingsPanel - Configurable settings modal for VTK viewers
 *
 * Generates and manages a settings panel with configurable fields.
 * Different viewers can specify which settings to show.
 */

/**
 * Available setting types
 */
export const SettingTypes = {
    NUMBER: 'number',
    COLOR: 'color',
    SELECT: 'select',
    CHECKBOX: 'checkbox'
};

/**
 * Default settings configuration
 */
const DEFAULT_SETTINGS_CONFIG = [
    {
        id: 'pointSize',
        label: 'Point Size',
        type: SettingTypes.NUMBER,
        min: 1,
        max: 20,
        step: 1,
        default: 5
    },
    {
        id: 'meshColor',
        label: 'Mesh Color',
        type: SettingTypes.COLOR,
        default: '#ffffff'
    },
    {
        id: 'edgeWidth',
        label: 'Edge Width',
        type: SettingTypes.NUMBER,
        min: 1,
        max: 10,
        step: 1,
        default: 1
    },
    {
        id: 'representation',
        label: 'Render Mode',
        type: SettingTypes.SELECT,
        options: [
            { value: '2', label: 'Surface' },
            { value: '1', label: 'Wireframe' },
            { value: '0', label: 'Points' }
        ],
        default: '2'
    },
    {
        id: 'parallelProjection',
        label: 'Parallel Projection',
        type: SettingTypes.CHECKBOX,
        default: false
    },
    {
        id: 'showAxisIndicator',
        label: 'Show Axis Indicator',
        type: SettingTypes.CHECKBOX,
        default: false
    }
];

/**
 * Colormap range settings (optional, for field visualization)
 */
const COLORMAP_SETTINGS_CONFIG = [
    {
        id: 'autoColorRange',
        label: 'Auto Range',
        type: SettingTypes.CHECKBOX,
        default: true,
        section: 'Colormap Range'
    },
    {
        id: 'colormapMin',
        label: 'Min',
        type: SettingTypes.NUMBER,
        step: 'any',
        placeholder: 'Auto',
        default: '',
        section: 'Colormap Range'
    },
    {
        id: 'colormapMax',
        label: 'Max',
        type: SettingTypes.NUMBER,
        step: 'any',
        placeholder: 'Auto',
        default: '',
        section: 'Colormap Range'
    }
];

export class SettingsPanel {
    /**
     * Create a SettingsPanel
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.container - Container element for the panel
     * @param {Function} options.onApply - Callback when settings are applied
     * @param {string[]} options.enabledSettings - Array of setting IDs to show (null = all)
     * @param {boolean} options.includeColormapSettings - Whether to include colormap range settings
     */
    constructor(options = {}) {
        this.container = options.container || document.body;
        this.onApply = options.onApply || (() => {});
        this.enabledSettings = options.enabledSettings || null;
        this.includeColormapSettings = options.includeColormapSettings || false;

        this.panelElement = null;
        this.overlayElement = null;
        this.currentValues = {};

        // Build settings config
        this.settingsConfig = this._buildSettingsConfig();

        // Initialize default values
        this.settingsConfig.forEach(setting => {
            this.currentValues[setting.id] = setting.default;
        });
    }

    /**
     * Build the settings configuration based on enabled settings
     * @private
     */
    _buildSettingsConfig() {
        let config = [...DEFAULT_SETTINGS_CONFIG];

        if (this.includeColormapSettings) {
            config = config.concat(COLORMAP_SETTINGS_CONFIG);
        }

        if (this.enabledSettings) {
            config = config.filter(s => this.enabledSettings.includes(s.id));
        }

        return config;
    }

    /**
     * Generate the HTML for the settings panel
     * @returns {string} HTML string
     */
    generateHTML() {
        let currentSection = null;
        let html = `
            <div id="settingsOverlay"></div>
            <div id="settingsPanel">
                <h3>Appearance Settings</h3>
        `;

        this.settingsConfig.forEach(setting => {
            // Add section header if needed
            if (setting.section && setting.section !== currentSection) {
                currentSection = setting.section;
                html += `
                    <div class="setting-row setting-section">
                        <label class="setting-section-label">${setting.section}:</label>
                    </div>
                `;
            }

            html += this._generateSettingRow(setting);
        });

        html += `
                <div class="button-row">
                    <button id="applySettings">Apply</button>
                    <button id="cancelSettings">Cancel</button>
                </div>
            </div>
        `;

        return html;
    }

    /**
     * Generate HTML for a single setting row
     * @private
     */
    _generateSettingRow(setting) {
        let inputHtml = '';

        switch (setting.type) {
            case SettingTypes.NUMBER:
                inputHtml = `
                    <input type="number" id="${setting.id}"
                        ${setting.min !== undefined ? `min="${setting.min}"` : ''}
                        ${setting.max !== undefined ? `max="${setting.max}"` : ''}
                        ${setting.step !== undefined ? `step="${setting.step}"` : ''}
                        ${setting.placeholder ? `placeholder="${setting.placeholder}"` : ''}
                        value="${setting.default}">
                `;
                break;

            case SettingTypes.COLOR:
                inputHtml = `<input type="color" id="${setting.id}" value="${setting.default}">`;
                break;

            case SettingTypes.SELECT:
                inputHtml = `<select id="${setting.id}">`;
                setting.options.forEach(opt => {
                    const selected = opt.value === setting.default ? 'selected' : '';
                    inputHtml += `<option value="${opt.value}" ${selected}>${opt.label}</option>`;
                });
                inputHtml += '</select>';
                break;

            case SettingTypes.CHECKBOX:
                const checked = setting.default ? 'checked' : '';
                return `
                    <div class="setting-row">
                        <label>
                            <input type="checkbox" id="${setting.id}" ${checked}> ${setting.label}
                        </label>
                    </div>
                `;
        }

        return `
            <div class="setting-row">
                <label>${setting.label}:</label>
                ${inputHtml}
            </div>
        `;
    }

    /**
     * Render the panel into the container
     */
    render() {
        // Create wrapper if needed
        const wrapper = document.createElement('div');
        wrapper.innerHTML = this.generateHTML();

        // Find or create panel elements in container
        this.overlayElement = wrapper.querySelector('#settingsOverlay');
        this.panelElement = wrapper.querySelector('#settingsPanel');

        // Append to container
        this.container.appendChild(this.overlayElement);
        this.container.appendChild(this.panelElement);

        // Setup event listeners
        this._setupEventListeners();
    }

    /**
     * Attach to existing panel elements in the DOM
     * @param {HTMLElement} panel - The settings panel element
     * @param {HTMLElement} overlay - The overlay element
     */
    attachToExisting(panel, overlay) {
        this.panelElement = panel;
        this.overlayElement = overlay;
        this._setupEventListeners();
    }

    /**
     * Setup event listeners
     * @private
     */
    _setupEventListeners() {
        // Apply button
        const applyBtn = this.panelElement.querySelector('#applySettings');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this._handleApply());
        }

        // Cancel button
        const cancelBtn = this.panelElement.querySelector('#cancelSettings');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.close());
        }

        // Overlay click
        if (this.overlayElement) {
            this.overlayElement.addEventListener('click', () => this.close());
        }
    }

    /**
     * Handle apply button click
     * @private
     */
    _handleApply() {
        const values = this.getValues();
        this.currentValues = { ...values };
        this.onApply(values);
        this.close();
    }

    /**
     * Open the settings panel
     */
    open() {
        // Update inputs with current values
        this.setValues(this.currentValues);

        if (this.panelElement) {
            this.panelElement.classList.add('visible');
        }
        if (this.overlayElement) {
            this.overlayElement.classList.add('visible');
        }
    }

    /**
     * Close the settings panel
     */
    close() {
        if (this.panelElement) {
            this.panelElement.classList.remove('visible');
        }
        if (this.overlayElement) {
            this.overlayElement.classList.remove('visible');
        }
    }

    /**
     * Get current values from all inputs
     * @returns {Object} Object with setting id -> value pairs
     */
    getValues() {
        const values = {};

        this.settingsConfig.forEach(setting => {
            const element = document.getElementById(setting.id);
            if (!element) return;

            if (setting.type === SettingTypes.CHECKBOX) {
                values[setting.id] = element.checked;
            } else {
                values[setting.id] = element.value;
            }
        });

        return values;
    }

    /**
     * Set values on all inputs
     * @param {Object} values - Object with setting id -> value pairs
     */
    setValues(values) {
        this.settingsConfig.forEach(setting => {
            const element = document.getElementById(setting.id);
            if (!element || values[setting.id] === undefined) return;

            if (setting.type === SettingTypes.CHECKBOX) {
                element.checked = values[setting.id];
            } else {
                element.value = values[setting.id];
            }
        });
    }

    /**
     * Update stored current values
     * @param {Object} values - Partial values to update
     */
    updateCurrentValues(values) {
        this.currentValues = { ...this.currentValues, ...values };
    }

    /**
     * Get current stored values
     * @returns {Object} Current values
     */
    getCurrentValues() {
        return { ...this.currentValues };
    }
}

export default SettingsPanel;
