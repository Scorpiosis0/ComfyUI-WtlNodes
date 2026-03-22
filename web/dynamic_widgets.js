import { app } from "../../scripts/app.js";

const HIDDEN_TAG = "tgszhidden";

// ============================================================================
// NODE CONFIGURATION - All node types and their widget configurations
// ============================================================================
const NODE_CONFIGS = {
    // Empty Latent node with special logic
    "EmptyLatent": {
        type: "special",
        setup: setupEmptyLatentLogic
    },
    
    // Interactive effect nodes with Apply/Skip buttons
    "DepthDOF": {
        type: "interactive",
        nodeType: "dof",
        widgets: ["focus_depth", "focus_range", "edge_fix", "hard_focus_range", "blur_strength"]
    },
    "CameraDepthDOF": {
        type: "interactive",
        nodeType: "cdof",
        widgets: ["focal_point", "focus_falloff", "edge_fix", "focal_plane", "blur_strength", "in_focus_mask_fix", "bokeh_shape", "highlight_factor", "highlight_threshold_low", "highlight_threshold_high", "depth_aware_blur", "blur_fixed_edge"]
    },
    "Saturation": {
        type: "interactive",
        nodeType: "sat",
        widgets: ["saturation"]
    },
    "Exposure": {
        type: "interactive",
        nodeType: "exp",
        widgets: ["exposure"]
    },
    "Contrast": {
        type: "interactive",
        nodeType: "con",
        widgets: ["contrast"]
    },
    "Brightness": {
        type: "interactive",
        nodeType: "bri",
        widgets: ["brightness"]
    },
    "Temperature": {
        type: "interactive",
        nodeType: "tem",
        widgets: ["temperature"]
    },
    "Hue": {
        type: "interactive",
        nodeType: "hue",
        widgets: ["hue"]
    },
    "HighlightShadow": {
        type: "interactive",
        nodeType: "has",
        widgets: ["shadow_adjustment", "highlight_adjustment", "midpoint", "feather_radius"]
    },
    "MaskProcessor": {
        type: "interactive",
        nodeType: "mpr",
        widgets: ["dilate_erode", "feather"]
    },
    "MaskFilter": {
        type: "interactive",
        nodeType: "mfl",
        widgets: ["area_x", "area_y", "keep"],
        setup: setupMaskFilterVisibility
    },
    "Dither": {
        type: "interactive",
        nodeType: "dit",
        widgets: ["dither_method", "r_levels", "g_levels", "b_levels", "dither_scale"]
    },
    "ASCII": {
        type: "interactive",
        nodeType: "asc",
        widgets: ["red_weight", "green_weight", "blue_weight", "char_set", "char_size", 
                  "background", "font_name", "bold", "italic", "spacing"]
    },
    "ImageTranslation": {
        type: "interactive",
        nodeType: "itr",
        widgets: ["translate_x", "translate_y", "bg_color"]
    },
    "ImageRotation": {
        type: "interactive",
        nodeType: "iro",
        widgets: ["rotate", "interpolation", "fit_mode", "bg_color"]
    },
    "ImageZoom": {
        type: "interactive",
        nodeType: "izo",
        widgets: ["zoom", "interpolation", "translate_x", "translate_y", "bg_color"]
    },
    "ImageResize": {
        type: "interactive",
        nodeType: "ire",
        widgets: ["resize_by", "width", "height", "multiplier", "interpolation", "fit_mode", "bg_color"],
        setup: setupImageResizeVisibility
    },
    "MaskTranslation": {
        type: "interactive",
        nodeType: "mtr",
        widgets: ["translate_x", "translate_y", "enhanced_visibility"]
    },
    "MaskRotation": {
        type: "interactive",
        nodeType: "mro",
        widgets: ["rotate", "interpolation", "fit_mode", "enhanced_visibility"]
    },
    "MaskZoom": {
        type: "interactive",
        nodeType: "mzo",
        widgets: ["zoom", "interpolation", "translate_x", "translate_y", "enhanced_visibility"]
    },
    "MaskResize": {
        type: "interactive",
        nodeType: "mre",
        widgets: ["resize_by", "width", "height", "multiplier", "interpolation", "fit_mode", "enhanced_visibility"],
        setup: setupMaskResizeVisibility
    },
    "FilmGrain": {
        type: "interactive",
        nodeType: "fgr",
        widgets: ["intensity", "grain_size", "monochrome"]
    },
    "ChromaticAberration": {
        type: "interactive",
        nodeType: "chromatic",
        widgets: ["offset_x", "offset_y", "red_scale", "blue_scale", "center_x", "center_y", "falloff"]
    },
    "FilmArtifacts": {
        type: "interactive",
        nodeType: "fil",
        widgets: ["intensity", "scratch_density", "scratch_max_length", "scratch_max_width", "dust_density", "dust_max_size", "hair_density", "hair_max_length", "light_leak_intensity", "vignette_strength", "seed"]
    },
    // NEW: Image Filters node with Apply Again button and conditional sliders
    "ImageFilters": {
        type: "interactive",
        nodeType: "iflt",
        widgets: ["filter_type", "strength", "edge_threshold", "neon_hue", "neon_blur"],
        hasApplyAgain: true,
        setup: setupImageFiltersVisibility
    },
    "CRTEffect": {
        type: "interactive",
        nodeType: "crt",
        widgets: ["scanline_intensity", "scanline_width", "curvature", "chromatic_aberration", "halation", "phosphor_dots", "noise", "vignette"]
    },
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const findWidgetByName = (node, name) => node.widgets?.find((w) => w.name === name);

function toggleWidget(node, widget, force) {
    if (!widget) return;
    
    widget.options[HIDDEN_TAG] ??= (widget.options.origType = widget.type, widget.options.origComputeSize = widget.computeSize, HIDDEN_TAG);
    const hide = force ?? (widget.type !== HIDDEN_TAG);
    
    widget.type = hide ? widget.options[HIDDEN_TAG] : widget.options.origType;
    widget.hidden = hide ? true : undefined;
    widget.computeSize = hide ? () => [0, -3.3] : widget.options.origComputeSize;
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, force));
    
    for (const el of ["inputEl", "input"])
        widget[el]?.classList?.toggle(HIDDEN_TAG, force);
    
    const height = hide ? node.size[1] : Math.max(node.computeSize()[1], node.size[1]);
    node.setSize([node.size[0], height]);
    widget.computedHeight = hide ? 0 : undefined;
}

function setupNodeRemovalHandler(node, nodeTypeKey) {
    const originalOnRemoved = node.onRemoved;
    
    node.onRemoved = function() {
        console.log(`[${node.comfyClass}] Node ${this.id} being removed, sending skip signal`);
        
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: this.id,
                node_type: nodeTypeKey,
                action: 'skip'
            })
        }).catch(err => console.error(`Failed to send skip signal for node ${this.id}:`, err));
        
        if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
    };
}

// ============================================================================
// GENERIC INTERACTIVE NODE SETUP
// ============================================================================

function setupInteractiveNode(node, config) {
    const { nodeType, widgets: widgetNames, hasApplyAgain } = config;
    
    // Find all widgets
    const widgets = widgetNames.map(name => findWidgetByName(node, name)).filter(Boolean);
    
    if (widgets.length === 0) return;
    
    // Function to send updated parameters
    const sendParams = () => {
        const params = { node_id: node.id, node_type: nodeType };
        widgetNames.forEach((name, i) => {
            if (widgets[i]) params[name] = widgets[i].value;
        });
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
    };
    
    // Attach callbacks to all widgets
    widgets.forEach(widget => {
        const origCallback = widget.callback;
        widget.callback = function(value) {
            sendParams();
            // Trigger processing status if available
            if (node.imageFilterStatus) {
                node.imageFilterStatus.showProcessing();
            }
            if (origCallback) origCallback.call(this, value);
        };
    });
    
    // Create button container
    const buttonContainer = document.createElement("div");
    buttonContainer.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 4px 8px 8px 8px;
        width: 100%;
    `;
    
    // Helper to create styled button
    const createStyledButton = (text, action, colors) => {
        const button = document.createElement("button");
        button.textContent = text;
        button.style.cssText = `
            padding: 4px 8px;
            background-color: ${colors.bg};
            color: #ffffff;
            border: 2px solid #3f3f46;
            border-radius: 2px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.15s;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        `;
        
        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = colors.hover;
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.backgroundColor = colors.bg;
        });
        
        button.addEventListener('mousedown', () => {
            button.style.backgroundColor = colors.active;
        });
        
        button.addEventListener('mouseup', () => {
            button.style.backgroundColor = colors.hover;
        });
        
        button.addEventListener('click', (e) => {
            e.stopPropagation();
            fetch('/tgsz_control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    node_id: node.id,
                    node_type: nodeType,
                    action: action
                })
            });
        });
        
        return button;
    };
    
    // Add Apply button
    const applyButton = createStyledButton('✓ Apply Effect', 'apply', {
        bg: '#52525b',
        hover: '#86a089',
        active: '#5a6f5c'
    });
    buttonContainer.appendChild(applyButton);
    
    // Add Apply Again button ONLY if specified in config
    if (hasApplyAgain) {
        const applyAgainButton = createStyledButton('↻ Apply Again', 'apply_again', {
            bg: '#52525b',
            hover: '#8b9fc4',
            active: '#5d7196'
        });
        buttonContainer.appendChild(applyAgainButton);
    }
    
    // Add Skip button
    const skipButton = createStyledButton('× Skip Effect', 'skip', {
        bg: '#52525b',
        hover: '#9ca3af',
        active: '#6b7280'
    });
    buttonContainer.appendChild(skipButton);
    
    // Add loading status bar
    const statusBar = document.createElement("div");
    statusBar.style.cssText = `
        padding: 4px 8px;
        background-color: #52525b;
        color: #ffffff;
        border: 2px solid #3f3f46;
        border-radius: 2px;
        font-size: 11px;
        font-weight: 500;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        text-align: center;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    statusBar.textContent = 'Ready';
    buttonContainer.appendChild(statusBar);
    
    // Status bar animation helper
    let statusDotCount = 0;
    let statusInterval = null;
    let pollingInterval = null;
    
    const showProcessing = () => {
        statusDotCount = 0;
        statusBar.textContent = 'Processing.';
        if (statusInterval) clearInterval(statusInterval);
        statusInterval = setInterval(() => {
            statusDotCount = (statusDotCount + 1) % 3;
            statusBar.textContent = 'Processing' + '.'.repeat(statusDotCount + 1);
        }, 500);
        
        // Start polling for completion
        if (pollingInterval) clearInterval(pollingInterval);
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`/tgsz_time/${nodeType}/${node.id}`);
                const data = await response.json();
                // Check if processing is complete
                if (data.status === 'ok' && data.complete === true) {
                    showDone(data.processing_time_ms);
                    if (pollingInterval) {
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                    }
                }
            } catch (err) {
                console.error('Failed to poll processing time:', err);
            }
        }, 100);  // Poll every 100ms
    };
    
    const showDone = (ms) => {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
        statusBar.textContent = `Done - ${ms}ms`;
    };
    
    const showReady = () => {
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
        statusBar.textContent = 'Ready';
    };
    
    // Store status functions on node for access from widget callbacks
    node.imageFilterStatus = { showProcessing, showDone, showReady };
    
    // Calculate button count for height
    // Each button: 4px top padding + ~23px button (with 2px border) + 4px bottom padding = ~31px
    // Gap between buttons: 4px
    // Status bar: 22px height + 4px gap
    // Container padding: 4px top + 8px bottom = 12px
    const buttonCount = hasApplyAgain ? 3 : 2;
    const buttonHeight = 31; // Approximate height per button including padding and 2px border
    const statusBarHeight = 22;
    const gapHeight = 4;
    const containerPadding = 12; // 4px top + 8px bottom
    const containerHeight = containerPadding + (buttonCount * buttonHeight) + ((buttonCount - 1) * gapHeight) + gapHeight + statusBarHeight;
    
    // Add the button container as a DOM widget
    const buttonWidget = node.addDOMWidget("buttons", "custom", buttonContainer, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return null; },
        setValue(v) {}
    });
    
    // Set computed size so ComfyUI knows how much space this widget takes
    buttonWidget.computeSize = function(width) {
        return [width, containerHeight];
    };
    
    // Setup removal handler
    setupNodeRemovalHandler(node, nodeType);
    
    // Call custom setup if provided
    if (config.setup) {
        config.setup(node);
    }
}

// ============================================================================
// SPECIAL NODE SETUPS (nodes with unique visibility logic)
// ============================================================================

function setupEmptyLatentLogic(node) {
    const useRatioWidget = findWidgetByName(node, "use_ratio");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const ratioWidget = findWidgetByName(node, "ratio");
    const resolutionWidget = findWidgetByName(node, "resolution");
    
    if (!useRatioWidget) return;
    
    const RESOLUTIONS = {
        "1:1": ["512x512","768x768","1024x1024","1536x1536","2048x2048"],
        "3:2": ["960x640","1152x768","1344x896","1536x1024","1728x1152","1920x1280"],
        "4:3": ["768x576","1024x768","1280x960","1536x1152","1792x1344"],
        "5:3": ["960x576","1280x768","1600x960","1920x1152"],
        "16:9": ["1024x576","2048x1152"],
        "16:10": ["1024x640","2048x1280"],
        "21:9": ["1344x576","2688x1152"],
        "32:9": ["2048x576","4096x1152"],
    };
    
    const updateResolutionOptions = () => {
        if (!ratioWidget || !resolutionWidget) return;
        const selectedRatio = ratioWidget.value;
        const newOptions = RESOLUTIONS[selectedRatio] || ["1024x1024"];
        resolutionWidget.options.values = newOptions;
        if (!newOptions.includes(resolutionWidget.value)) {
            resolutionWidget.value = newOptions[0];
        }
        if (resolutionWidget.callback) resolutionWidget.callback(resolutionWidget.value);
    };
    
    const updateVisibility = () => {
        const useRatio = useRatioWidget.value;
        toggleWidget(node, widthWidget, useRatio);
        toggleWidget(node, heightWidget, useRatio);
        toggleWidget(node, ratioWidget, !useRatio);
        toggleWidget(node, resolutionWidget, !useRatio);
        node.setDirtyCanvas(true);
    };
    
    updateVisibility();
    updateResolutionOptions();
    
    let val = useRatioWidget.value;
    Object.defineProperty(useRatioWidget, 'value', {
        get() { return val; },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
    
    let ratioVal = ratioWidget.value;
    Object.defineProperty(ratioWidget, 'value', {
        get() { return ratioVal; },
        set(newVal) {
            if (newVal !== ratioVal) {
                ratioVal = newVal;
                updateResolutionOptions();
            }
        }
    });
}

function setupMaskFilterVisibility(node) {
    const areaYWidget = findWidgetByName(node, "area_y");
    const keepWidget = findWidgetByName(node, "keep");
    
    const updateKeepVisibility = () => {
        if (!keepWidget) return;
        const shouldShow = keepWidget.value !== "between_x_y";
        toggleWidget(node, areaYWidget, shouldShow);
        node.setDirtyCanvas(true);
    };
    
    updateKeepVisibility();
    
    if (keepWidget) {
        let keepVal = keepWidget.value;
        Object.defineProperty(keepWidget, "value", {
            get() { return keepVal; },
            set(newVal) {
                if (newVal !== keepVal) {
                    keepVal = newVal;
                    updateKeepVisibility();
                    if (keepWidget.callback) keepWidget.callback.call(this, newVal);
                }
            }
        });
    }
}

function setupImageResizeVisibility(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    
    const updateVisibility = () => {
        const resizeBy = resizeByWidget.value;
        toggleWidget(node, widthWidget, resizeBy);
        toggleWidget(node, heightWidget, resizeBy);
        toggleWidget(node, fitModeWidget, resizeBy);
        toggleWidget(node, multiplierWidget, !resizeBy);
        node.setDirtyCanvas(true);
    };
    
    updateVisibility();
    
    let val = resizeByWidget.value;
    Object.defineProperty(resizeByWidget, 'value', {
        get() { return val; },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
}

function setupMaskResizeVisibility(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    
    const updateVisibility = () => {
        const resizeBy = resizeByWidget.value;
        toggleWidget(node, widthWidget, resizeBy);
        toggleWidget(node, heightWidget, resizeBy);
        toggleWidget(node, fitModeWidget, resizeBy);
        toggleWidget(node, multiplierWidget, !resizeBy);
        node.setDirtyCanvas(true);
    };
    
    updateVisibility();
    
    let val = resizeByWidget.value;
    Object.defineProperty(resizeByWidget, 'value', {
        get() { return val; },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
}

function setupImageFiltersVisibility(node) {
    const filterTypeWidget = findWidgetByName(node, "filter_type");
    const edgeThresholdWidget = findWidgetByName(node, "edge_threshold");
    const neonHueWidget = findWidgetByName(node, "neon_hue");
    const neonBlurWidget = findWidgetByName(node, "neon_blur");
    
    const updateVisibility = () => {
        const filterType = filterTypeWidget.value;
        
        // Show edge_threshold for cartoon and neon
        const showEdgeThreshold = (filterType === "cartoon" || filterType === "neon");
        toggleWidget(node, edgeThresholdWidget, !showEdgeThreshold);
        
        // Show neon_hue and neon_blur only for neon
        const showNeonParams = (filterType === "neon");
        toggleWidget(node, neonHueWidget, !showNeonParams);
        toggleWidget(node, neonBlurWidget, !showNeonParams);
        
        node.setDirtyCanvas(true);
    };
    
    updateVisibility();
    
    let val = filterTypeWidget.value;
    Object.defineProperty(filterTypeWidget, 'value', {
        get() { return val; },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
}

// ============================================================================
// REGISTER EXTENSION
// ============================================================================

app.registerExtension({
    name: "Comfy.TgszNodes.DynamicWidgets",
    
    async init() {
        const style = document.createElement("style");
        style.textContent = `
            .${HIDDEN_TAG} {
                display: none !important;
            }
            
            /* Target ComfyUI button widgets specifically */
            button.tgsz-button,
            input[type="button"].tgsz-button {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                padding: 8px 16px !important;
                border-radius: 4px !important;
                cursor: pointer !important;
                transition: background-color 0.15s ease, border-color 0.15s ease !important;
                outline: none !important;
                letter-spacing: 0.01em !important;
                background-color: #52525b !important;
                border: 2px solid #3f3f46 !important;
                color: #ffffff !important;
            }
            
            /* Apply Button - Subtle Green on hover/active */
            button.tgsz-apply:hover,
            input[type="button"].tgsz-apply:hover {
                background-color: #86a089 !important;
                border-color: #5a6f5c !important;
            }
            
            button.tgsz-apply:active,
            input[type="button"].tgsz-apply:active {
                background-color: #5a6f5c !important;
                border-color: #3d4a3e !important;
            }
            
            /* Apply Again Button - Subtle Blue on hover/active */
            button.tgsz-apply-again:hover,
            input[type="button"].tgsz-apply-again:hover {
                background-color: #8b9fc4 !important;
                border-color: #5d7196 !important;
            }
            
            button.tgsz-apply-again:active,
            input[type="button"].tgsz-apply-again:active {
                background-color: #5d7196 !important;
                border-color: #3e4d64 !important;
            }
            
            /* Skip Button - Subtle Gray on hover/active */
            button.tgsz-skip:hover,
            input[type="button"].tgsz-skip:hover {
                background-color: #9ca3af !important;
                border-color: #6b7280 !important;
            }
            
            button.tgsz-skip:active,
            input[type="button"].tgsz-skip:active {
                background-color: #6b7280 !important;
                border-color: #4b5563 !important;
            }
            
            /* Button Icons */
            .tgsz-button-icon {
                display: inline-block;
                margin-right: 6px;
                font-size: 14px;
                vertical-align: middle;
            }
            
            /* Fix skip emoji vertical alignment */
            .tgsz-skip .tgsz-button-icon {
                position: relative;
                top: -1px;
            }
        `;
        document.head.appendChild(style);
    },
    
    async nodeCreated(node) {
        const config = NODE_CONFIGS[node.comfyClass];
        
        if (!config) return;
        
        if (config.type === "interactive") {
            setupInteractiveNode(node, config);
        } else if (config.type === "special" && config.setup) {
            config.setup(node);
        }
    }
});