/**
 * ControlsBar - Configurable controls bar for VTK viewers
 *
 * Generates and manages the bottom controls bar with camera buttons,
 * toggles, selectors, and action buttons.
 */

/**
 * Control types
 */
export const ControlTypes = {
    BUTTON: 'button',
    CHECKBOX: 'checkbox',
    SELECT: 'select',
    DIVIDER: 'divider',
    LABEL: 'label'
};

/**
 * Standard control configurations
 */
export const StandardControls = {
    settingsButton: {
        id: 'settingsButton',
        type: ControlTypes.BUTTON,
        label: '[Settings]',
        title: 'Appearance Settings'
    },
    divider: {
        type: ControlTypes.DIVIDER
    },
    edgesToggle: {
        id: 'showEdges',
        type: ControlTypes.CHECKBOX,
        label: 'Edges'
    },
    cameraButtons: {
        type: 'camera_group',
        buttons: ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    },
    screenshotButton: {
        id: 'screenshot',
        type: ControlTypes.BUTTON,
        label: 'Screenshot'
    },
    saveMeshButton: {
        id: 'saveMesh',
        type: ControlTypes.BUTTON,
        label: 'Save Mesh',
        className: 'save-button'
    },
    saveGLBButton: {
        id: 'saveGLB',
        type: ControlTypes.BUTTON,
        label: 'Save GLB',
        className: 'save-button'
    },
    resetCameraButton: {
        id: 'resetCamera',
        type: ControlTypes.BUTTON,
        label: 'Reset Camera'
    },
    fieldSelector: {
        id: 'fieldSelector',
        type: ControlTypes.SELECT,
        label: 'Field',
        wrapperClass: 'field-label',
        hidden: true
    },
    colormapSelector: {
        id: 'colormapSelector',
        type: ControlTypes.SELECT,
        label: 'Colormap',
        wrapperClass: 'colormap-label',
        hidden: true
    }
};

export class ControlsBar {
    /**
     * Create a ControlsBar
     * @param {Object} options - Configuration options
     * @param {HTMLElement} options.container - Container element for the controls bar
     * @param {Object} options.handlers - Event handlers keyed by control ID
     * @param {Array} options.controls - Array of control configurations
     */
    constructor(options = {}) {
        this.container = options.container;
        this.handlers = options.handlers || {};
        this.controls = options.controls || this._getDefaultControls();
        this.elements = {};
    }

    /**
     * Get default controls for standard viewer
     * @private
     */
    _getDefaultControls() {
        return [
            StandardControls.settingsButton,
            StandardControls.divider,
            StandardControls.edgesToggle,
            StandardControls.divider,
            StandardControls.cameraButtons,
            StandardControls.divider,
            StandardControls.fieldSelector,
            StandardControls.colormapSelector,
            StandardControls.screenshotButton,
            StandardControls.saveMeshButton
        ];
    }

    /**
     * Generate HTML for the controls bar
     * @returns {string} HTML string
     */
    generateHTML() {
        let html = '';

        this.controls.forEach(control => {
            if (!control) return;

            switch (control.type) {
                case ControlTypes.BUTTON:
                    html += this._generateButton(control);
                    break;
                case ControlTypes.CHECKBOX:
                    html += this._generateCheckbox(control);
                    break;
                case ControlTypes.SELECT:
                    html += this._generateSelect(control);
                    break;
                case ControlTypes.DIVIDER:
                    html += '<span class="controls-divider"></span>';
                    break;
                case ControlTypes.LABEL:
                    html += `<span class="controls-label">${control.text || ''}</span>`;
                    break;
                case 'camera_group':
                    html += this._generateCameraGroup(control);
                    break;
                default:
                    if (typeof control.html === 'string') {
                        html += control.html;
                    }
            }
        });

        return html;
    }

    /**
     * Generate button HTML
     * @private
     */
    _generateButton(control) {
        const className = control.className || '';
        const style = control.id === 'settingsButton' ? 'width:32px;height:32px;padding:0;font-size:16px;' : '';
        const styleAttr = style ? `style="${style}"` : '';
        const classAttr = className ? `class="${className}"` : '';

        return `<button id="${control.id}" ${classAttr} ${styleAttr} title="${control.title || ''}">${control.label}</button>`;
    }

    /**
     * Generate checkbox HTML
     * @private
     */
    _generateCheckbox(control) {
        const checked = control.checked ? 'checked' : '';
        return `
            <label>
                <input type="checkbox" id="${control.id}" ${checked}> ${control.label}
            </label>
        `;
    }

    /**
     * Generate select HTML
     * @private
     */
    _generateSelect(control) {
        const wrapperClass = control.wrapperClass || '';
        const hiddenStyle = control.hidden ? 'style="display:none;"' : '';
        const options = (control.options || []).map(opt =>
            `<option value="${opt.value}" ${opt.selected ? 'selected' : ''}>${opt.label}</option>`
        ).join('');

        return `
            <label id="${control.id}Label" class="${wrapperClass}" ${hiddenStyle}>
                ${control.label}:
                <select id="${control.id}">${options}</select>
            </label>
        `;
    }

    /**
     * Generate camera view buttons group
     * @private
     */
    _generateCameraGroup(control) {
        const buttons = control.buttons || ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
        return buttons.map(dir => {
            const id = `view${dir.replace('+', 'Pos').replace('-', 'Neg')}`;
            return `<button id="${id}">${dir}</button>`;
        }).join('');
    }

    /**
     * Render the controls bar into the container
     */
    render() {
        if (!this.container) {
            console.warn('[ControlsBar] No container specified');
            return;
        }

        this.container.innerHTML = this.generateHTML();
        this._cacheElements();
        this._setupEventListeners();
    }

    /**
     * Cache references to control elements
     * @private
     */
    _cacheElements() {
        this.controls.forEach(control => {
            if (control && control.id) {
                this.elements[control.id] = document.getElementById(control.id);
            }
        });

        // Also cache camera buttons
        ['viewPosX', 'viewNegX', 'viewPosY', 'viewNegY', 'viewPosZ', 'viewNegZ'].forEach(id => {
            const el = document.getElementById(id);
            if (el) this.elements[id] = el;
        });
    }

    /**
     * Setup event listeners based on handlers
     * @private
     */
    _setupEventListeners() {
        // Standard handlers
        Object.keys(this.handlers).forEach(id => {
            const element = this.elements[id];
            const handler = this.handlers[id];

            if (element && handler) {
                if (element.type === 'checkbox') {
                    element.addEventListener('change', (e) => handler(e.target.checked, e));
                } else if (element.tagName === 'SELECT') {
                    element.addEventListener('change', (e) => handler(e.target.value, e));
                } else {
                    element.addEventListener('click', handler);
                }
            }
        });

        // Camera buttons (special handling)
        if (this.handlers.cameraView) {
            const cameraDirections = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'];
            cameraDirections.forEach(dir => {
                const id = `view${dir.replace('+', 'Pos').replace('-', 'Neg')}`;
                const element = this.elements[id];
                if (element) {
                    element.addEventListener('click', () => this.handlers.cameraView(dir));
                }
            });
        }
    }

    /**
     * Get an element by ID
     * @param {string} id - Element ID
     * @returns {HTMLElement|null}
     */
    getElement(id) {
        return this.elements[id] || document.getElementById(id);
    }

    /**
     * Show a control element
     * @param {string} id - Element ID
     */
    show(id) {
        const wrapper = document.getElementById(`${id}Label`) || this.elements[id];
        if (wrapper) {
            wrapper.style.display = 'flex';
        }
    }

    /**
     * Hide a control element
     * @param {string} id - Element ID
     */
    hide(id) {
        const wrapper = document.getElementById(`${id}Label`) || this.elements[id];
        if (wrapper) {
            wrapper.style.display = 'none';
        }
    }

    /**
     * Enable a control element
     * @param {string} id - Element ID
     */
    enable(id) {
        const element = this.elements[id];
        if (element) {
            element.disabled = false;
        }
    }

    /**
     * Disable a control element
     * @param {string} id - Element ID
     */
    disable(id) {
        const element = this.elements[id];
        if (element) {
            element.disabled = true;
        }
    }

    /**
     * Set button text
     * @param {string} id - Button ID
     * @param {string} text - New text
     */
    setButtonText(id, text) {
        const element = this.elements[id];
        if (element) {
            element.textContent = text;
        }
    }

    /**
     * Set button style (e.g., for save button states)
     * @param {string} id - Button ID
     * @param {Object} style - Style properties
     */
    setButtonStyle(id, style) {
        const element = this.elements[id];
        if (element) {
            Object.assign(element.style, style);
        }
    }

    /**
     * Populate a select element with options
     * @param {string} id - Select element ID
     * @param {Array} options - Array of {value, label, selected?} objects
     * @param {boolean} clearFirst - Whether to clear existing options first
     */
    populateSelect(id, options, clearFirst = true) {
        const element = this.elements[id] || document.getElementById(id);
        if (!element || element.tagName !== 'SELECT') return;

        if (clearFirst) {
            element.innerHTML = '';
        }

        options.forEach(opt => {
            const option = document.createElement('option');
            option.value = opt.value;
            option.textContent = opt.label;
            if (opt.selected) option.selected = true;
            if (opt.disabled) option.disabled = true;
            element.appendChild(option);
        });
    }

    /**
     * Add a handler for a control
     * @param {string} id - Control ID
     * @param {Function} handler - Event handler function
     */
    addHandler(id, handler) {
        this.handlers[id] = handler;

        // If already rendered, attach immediately
        const element = this.elements[id];
        if (element) {
            if (element.type === 'checkbox') {
                element.addEventListener('change', (e) => handler(e.target.checked, e));
            } else if (element.tagName === 'SELECT') {
                element.addEventListener('change', (e) => handler(e.target.value, e));
            } else {
                element.addEventListener('click', handler);
            }
        }
    }
}

export default ControlsBar;
