import { app } from "../../scripts/app.js";

const HIDDEN_TAG = "tgszhidden";

// Helper

function setupNodeRemovalHandler(node, nodeTypeKey) {
    const originalOnRemoved = node.onRemoved;
    
    node.onRemoved = function() {
        console.log(`[${node.comfyClass}] Node ${this.id} being removed, sending skip signal`);
        
        // Send skip signal to Python backend
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: this.id,
                node_type: nodeTypeKey,
                action: 'skip'
            })
        }).catch(err => {
            console.error(`Failed to send skip signal for node ${this.id}:`, err);
        });
        
        // Call original onRemoved if it exists
        if (originalOnRemoved) {
            originalOnRemoved.apply(this, arguments);
        }
    };
}

const findWidgetByName = (node, name) => node.widgets?.find((w) => w.name === name);

export function toggleWidget(node, widget, force) {
    if (!widget) return;
    
    // Store original properties on first call
    widget.options[HIDDEN_TAG] ??= (widget.options.origType = widget.type, widget.options.origComputeSize = widget.computeSize, HIDDEN_TAG);
    
    const hide = force ?? (widget.type !== HIDDEN_TAG);
    
    widget.type = hide ? widget.options[HIDDEN_TAG] : widget.options.origType;
    
    if (hide) {
        widget.hidden = true;
    } else {
        delete widget.hidden;
    }
    
    widget.computeSize = hide ? () => [0, -3.3] : widget.options.origComputeSize;
    
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, force));
    
    for (const el of ["inputEl", "input"])
        widget[el]?.classList?.toggle(HIDDEN_TAG, force);
    
    const height = hide ? node.size[1] : Math.max(node.computeSize()[1], node.size[1]);
    node.setSize([node.size[0], height]);
    
    if (hide)
        widget.computedHeight = 0;
    else
        delete widget.computedHeight;
}

//Widget logic
function setupEmptyLatentLogic(node) {

    const useRatioWidget = findWidgetByName(node, "use_ratio");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const ratioWidget = findWidgetByName(node, "ratio");
    const resolutionWidget = findWidgetByName(node, "resolution");
    
    if (!useRatioWidget) return;
    
    // Resolution options for each ratio (must match Python RESOLUTIONS dict)
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
        
        // Update the widget's options
        resolutionWidget.options.values = newOptions;
        
        // Reset to first option if current value isn't in new options
        if (!newOptions.includes(resolutionWidget.value)) {
            resolutionWidget.value = newOptions[0];
        }
        
        // Force widget to redraw
        if (resolutionWidget.callback) {
            resolutionWidget.callback(resolutionWidget.value);
        }
    };
    
    const updateVisibility = () => {
        const useRatio = useRatioWidget.value;
        
        toggleWidget(node, widthWidget, useRatio);
        toggleWidget(node, heightWidget, useRatio);
        toggleWidget(node, ratioWidget, !useRatio);
        toggleWidget(node, resolutionWidget, !useRatio);
        
        node.setDirtyCanvas(true);
    };
    
    // Initial setup
    updateVisibility();
    updateResolutionOptions();
    
    // Watch for use_ratio changes
    let val = useRatioWidget.value;
    Object.defineProperty(useRatioWidget, 'value', {
        get() {
            return val;
        },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
    
    // Watch for ratio changes to update resolution dropdown
    let ratioVal = ratioWidget.value;
    Object.defineProperty(ratioWidget, 'value', {
        get() {
            return ratioVal;
        },
        set(newVal) {
            if (newVal !== ratioVal) {
                ratioVal = newVal;
                updateResolutionOptions();
            }
        }
    });
}

function setupDOFControls(node) {

    const focusDepthWidget = findWidgetByName(node, "focus_depth");
    const focusRangeWidget = findWidgetByName(node, "focus_range");
    const edgeFixWidget = findWidgetByName(node, "edge_fix");
    const hardFocusRangeWidget = findWidgetByName(node, "hard_focus_range");
    const blurStrengthWidget = findWidgetByName(node, "blur_strength");
    
    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!focusDepthWidget || !focusRangeWidget || !edgeFixWidget || !hardFocusRangeWidget || !blurStrengthWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dof',
                focus_depth: focusDepthWidget.value,
                focus_range: focusRangeWidget.value,
                edge_fix: edgeFixWidget.value,
                hard_focus_range : hardFocusRangeWidget.value,
                blur_strength : blurStrengthWidget.value
            })
        });
    };
    
    [focusDepthWidget, focusRangeWidget, edgeFixWidget, hardFocusRangeWidget, blurStrengthWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dof',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dof',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'dof');
}

function setupCamDOFControls(node) {

    const focusDepthWidget = findWidgetByName(node, "focus_depth");
    const focusRangeWidget = findWidgetByName(node, "focus_range");
    const edgeFixWidget = findWidgetByName(node, "edge_fix");
    const hardFocusRangeWidget = findWidgetByName(node, "hard_focus_range");
    const blurStrengthWidget = findWidgetByName(node, "blur_strength");
    
    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!focusDepthWidget || !focusRangeWidget || !edgeFixWidget || !hardFocusRangeWidget || !blurStrengthWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'cdof',
                focus_depth: focusDepthWidget.value,
                focus_range: focusRangeWidget.value,
                edge_fix: edgeFixWidget.value,
                hard_focus_range : hardFocusRangeWidget.value,
                blur_strength : blurStrengthWidget.value
            })
        });
    };
    
    [focusDepthWidget, focusRangeWidget, edgeFixWidget, hardFocusRangeWidget, blurStrengthWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'cdof',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'cdof',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'dof');
}

function setupSaturationControls(node) {

    const saturationWidget = findWidgetByName(node, "saturation");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!saturationWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'sat',
                saturation: saturationWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (saturationWidget) {
        const origCallback = saturationWidget.callback;
        saturationWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'sat',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'sat',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'sat');
}

function setupExposureControls(node) {

    const exposureWidget = findWidgetByName(node, "exposure");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!exposureWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'exp',
                exposure: exposureWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (exposureWidget) {
        const origCallback = exposureWidget.callback;
        exposureWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'exp',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'exp',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'exp');
}

function setupHighlightShadowControls(node) {

    const shadowWidget = findWidgetByName(node, "shadow_adjustment");
    const highlightWidget = findWidgetByName(node, "highlight_adjustment");
    const midpointWidget = findWidgetByName(node, "midpoint");
    const featherWidget = findWidgetByName(node, "feather_radius");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!shadowWidget || !highlightWidget || !midpointWidget || !featherWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'has',
                shadow_adjustment: shadowWidget.value,
                highlight_adjustment: highlightWidget.value,
                midpoint: midpointWidget.value,
                feather_radius: featherWidget.value,
            })
        });
    };
    
    [shadowWidget, highlightWidget, midpointWidget, featherWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'has',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'has',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'exp');
}

function setupMaskFilterControls(node) {

    const areaXWidget = findWidgetByName(node, "area_x");
    const areaYWidget = findWidgetByName(node, "area_y");
    const keepWidget = findWidgetByName(node, "keep");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!areaXWidget || !areaYWidget || !keepWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mfl',
                area_x: areaXWidget.value,
                area_y: areaYWidget.value,
                keep: keepWidget.value
            })
        });
    };
    
    [areaXWidget, areaYWidget, keepWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mfl',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mfl',
                action: 'skip'
            })
        });
    }, { serialize: false });

    // --- Widget visibility logic ---
    const updateKeepVisibility = () => {
        if (!keepWidget) return;
        
        const shouldShow = keepWidget.value !== "between_x_y";
        toggleWidget(node, areaYWidget, shouldShow);
        node.setDirtyCanvas(true);
    };

    // Initial visibility update
    updateKeepVisibility();

    // Watch for value changes on keepWidget
    if (keepWidget) {
        let keepVal = keepWidget.value;

        Object.defineProperty(keepWidget, "value", {
            get() {
                return keepVal;
            },
            set(newVal) {
                if (newVal !== keepVal) {
                    keepVal = newVal;

                    // Update visibility dynamically
                    updateKeepVisibility();

                    // Call original callback
                    if (keepWidget.callback)
                        keepWidget.callback.call(this, newVal);
                }
            }
        });
    }
    
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'mpr');
}

function setupMaskProcessorControls(node) {

    const dilateErodeWidget = findWidgetByName(node, "dilate_erode");
    const featherWidget = findWidgetByName(node, "feather");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!dilateErodeWidget || !featherWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mpr',
                dilate_erode: dilateErodeWidget.value,
                feather: featherWidget.value,
            })
        });
    };
    
    [dilateErodeWidget, featherWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mpr',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mpr',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'mpr');
}

function setupHueControls(node) {

    const hueWidget = findWidgetByName(node, "hue");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!hueWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'hue',
                hue: hueWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (hueWidget) {
        const origCallback = hueWidget.callback;
        hueWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'hue',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'hue',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'hue');
}

function setupContrastControls(node) {

    const contrastWidget = findWidgetByName(node, "contrast");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!contrastWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'con',
                contrast: contrastWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (contrastWidget) {
        const origCallback = contrastWidget.callback;
        contrastWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'con',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'con',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'con');
}

function setupTemperatureControls(node) {

    const temperatureWidget = findWidgetByName(node, "temperature");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!temperatureWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'tem',
                temperature: temperatureWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (temperatureWidget) {
        const origCallback = temperatureWidget.callback;
        temperatureWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'tem',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'tem',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'tem');
}

function setupBrightnessControls(node) {

    const brightnessWidget = findWidgetByName(node, "brightness");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!brightnessWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'bri',
                brightness: brightnessWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (brightnessWidget) {
        const origCallback = brightnessWidget.callback;
        brightnessWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'bri',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'bri',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'bri');
}

function setupDitheringControls(node) {

    const ditherMethodWidget = findWidgetByName(node, "dither_method");
    const rLevelsWidget = findWidgetByName(node, "r_levels");
    const gLevelsWidget = findWidgetByName(node, "g_levels");
    const bLevelsWidget = findWidgetByName(node, "b_levels");
    const ditherScaleWidget = findWidgetByName(node, "dither_scale");

    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!ditherMethodWidget || !rLevelsWidget || !gLevelsWidget || !bLevelsWidget || !ditherScaleWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dit',
                dither_method: ditherMethodWidget.value,
                r_levels: rLevelsWidget.value,
                g_levels: gLevelsWidget.value,
                b_levels: bLevelsWidget.value,
                dither_scale: ditherScaleWidget.value,
            })
        });
    };
    
    // Watch for slider changes and send updated params
    [ditherMethodWidget, rLevelsWidget, gLevelsWidget, bLevelsWidget, ditherScaleWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    // Add "Apply Effect" button
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        // Create flag file to signal Python to exit loop
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dit',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    // Add "Skip Effect" button  
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        // Create flag file to signal Python to skip effect
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dit',
                action: 'skip'
            })
        });
    }, { serialize: false });
    // Setup node removal handler to cancel on node deletion
    setupNodeRemovalHandler(node, 'dit');
}

function setupImageTranslationControls(node) {
    const translateXWidget = findWidgetByName(node, "translate_x");
    const translateYWidget = findWidgetByName(node, "translate_y");
    const bgColorWidget = findWidgetByName(node, "bg_color");

    const sendParams = () => {
        if (!translateXWidget || !translateYWidget || !bgColorWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'itr',
                translate_x: translateXWidget.value,
                translate_y: translateYWidget.value,
                bg_color: bgColorWidget.value,
            })
        });
    };
    
    [translateXWidget, translateYWidget, bgColorWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'itr',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'itr',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'itr');
}

function setupImageRotationControls(node) {
    const rotateWidget = findWidgetByName(node, "rotate");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const bgColorWidget = findWidgetByName(node, "bg_color");

    const sendParams = () => {
        if (!rotateWidget || !interpolationWidget || !fitModeWidget || !bgColorWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'iro',
                rotate: rotateWidget.value,
                interpolation: interpolationWidget.value,
                fit_mode: fitModeWidget.value,
                bg_color: bgColorWidget.value,
            })
        });
    };
    
    [rotateWidget, interpolationWidget, fitModeWidget, bgColorWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'iro',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'iro',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'iro');
}

function setupImageZoomControls(node) {
    const zoomWidget = findWidgetByName(node, "zoom");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const translateXWidget = findWidgetByName(node, "translate_x");
    const translateYWidget = findWidgetByName(node, "translate_y");
    const bgColorWidget = findWidgetByName(node, "bg_color");

    const sendParams = () => {
        if (!zoomWidget || !interpolationWidget || !translateXWidget || !translateYWidget || !bgColorWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'izo',
                zoom: zoomWidget.value,
                interpolation: interpolationWidget.value,
                translate_x: translateXWidget.value,
                translate_y: translateYWidget.value,
                bg_color: bgColorWidget.value,
            })
        });
    };
    
    [zoomWidget, interpolationWidget, translateXWidget, translateYWidget, bgColorWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'izo',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'izo',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'izo');
}

function setupImageResizeControls(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const bgColorWidget = findWidgetByName(node, "bg_color");

    const sendParams = () => {
        if (!resizeByWidget || !widthWidget || !heightWidget || !multiplierWidget || !interpolationWidget || !fitModeWidget || !bgColorWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'ire',
                resize_by: resizeByWidget.value,
                width: widthWidget.value,
                height: heightWidget.value,
                multiplier: multiplierWidget.value,
                interpolation: interpolationWidget.value,
                fit_mode: fitModeWidget.value,
                bg_color: bgColorWidget.value,
            })
        });
    };
    
    // Watch for changes on all widgets
    [resizeByWidget, widthWidget, heightWidget, multiplierWidget, interpolationWidget, fitModeWidget, bgColorWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'ire',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'ire',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    // Widget visibility logic
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
        get() {
            return val;
        },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
    
    setupNodeRemovalHandler(node, 'ire');
}

function setupMaskTranslationControls(node) {
    const translateXWidget = findWidgetByName(node, "translate_x");
    const translateYWidget = findWidgetByName(node, "translate_y");
    const enhancedVisibilityWidget = findWidgetByName(node, "enhanced_visibility");

    const sendParams = () => {
        if (!translateXWidget || !translateYWidget || !enhancedVisibilityWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mtr',
                translate_x: translateXWidget.value,
                translate_y: translateYWidget.value,
                enhanced_visibility: enhancedVisibilityWidget.value,
            })
        });
    };
    
    [translateXWidget, translateYWidget, enhancedVisibilityWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mtr',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mtr',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'mtr');
}

function setupMaskRotationControls(node) {
    const rotateWidget = findWidgetByName(node, "rotate");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const enhancedVisibilityWidget = findWidgetByName(node, "enhanced_visibility");

    const sendParams = () => {
        if (!rotateWidget || !interpolationWidget || !fitModeWidget || !enhancedVisibilityWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mro',
                rotate: rotateWidget.value,
                interpolation: interpolationWidget.value,
                fit_mode: fitModeWidget.value,
                enhanced_visibility: enhancedVisibilityWidget.value,
            })
        });
    };
    
    // Watch for changes on all widgets
    [rotateWidget, interpolationWidget, fitModeWidget, enhancedVisibilityWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mro',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mro',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'mro');
}

function setupMaskZoomControls(node) {
    const zoomWidget = findWidgetByName(node, "zoom");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const translateXWidget = findWidgetByName(node, "translate_x");
    const translateYWidget = findWidgetByName(node, "translate_y");
    const enhancedVisibilityWidget = findWidgetByName(node, "enhanced_visibility");

    const sendParams = () => {
        if (!zoomWidget || !interpolationWidget || !translateXWidget || !translateYWidget || !enhancedVisibilityWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mzo',
                zoom: zoomWidget.value,
                interpolation: interpolationWidget.value,
                translate_x: translateXWidget.value,
                translate_y: translateYWidget.value,
                enhanced_visibility: enhancedVisibilityWidget.value,
            })
        });
    };
    
    // Watch for changes on all widgets
    [zoomWidget, interpolationWidget, translateXWidget, translateYWidget, enhancedVisibilityWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mzo',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mzo',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    setupNodeRemovalHandler(node, 'mzo');
}

function setupMaskResizeControls(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    const interpolationWidget = findWidgetByName(node, "interpolation");
    const fitModeWidget = findWidgetByName(node, "fit_mode");
    const enhancedVisibilityWidget = findWidgetByName(node, "enhanced_visibility");

    const sendParams = () => {
        if (!resizeByWidget || !widthWidget || !heightWidget || !multiplierWidget || !interpolationWidget || !fitModeWidget || !enhancedVisibilityWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mre',
                resize_by: resizeByWidget.value,
                width: widthWidget.value,
                height: heightWidget.value,
                multiplier: multiplierWidget.value,
                interpolation: interpolationWidget.value,
                fit_mode: fitModeWidget.value,
                enhanced_visibility: enhancedVisibilityWidget.value,
            })
        });
    };
    
    // Watch for changes on all widgets
    [resizeByWidget, widthWidget, heightWidget, multiplierWidget, interpolationWidget, fitModeWidget, enhancedVisibilityWidget].forEach(widget => {
        if (widget) {
            const origCallback = widget.callback;
            widget.callback = function(value) {
                sendParams();
                if (origCallback) origCallback.call(this, value);
            };
        }
    });
    
    const applyButton = node.addWidget("button", "✅ Apply Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mre',
                action: 'apply'
            })
        });
    }, { serialize: false });
    
    const skipButton = node.addWidget("button", "⏭️ Skip Effect", null, () => {
        fetch('/tgsz_control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'mre',
                action: 'skip'
            })
        });
    }, { serialize: false });
    
    // Widget visibility logic
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
        get() {
            return val;
        },
        set(newVal) {
            if (newVal !== val) {
                val = newVal;
                updateVisibility();
            }
        }
    });
    
    setupNodeRemovalHandler(node, 'mre');
}

// Register
app.registerExtension({
    name: "Comfy.TgszNodes.DynamicWidgets",
    
    async init() {
        // Add CSS for hiding widgets
        const style = document.createElement("style");
        style.textContent = `.${HIDDEN_TAG} {display: none !important;}`;
        document.head.appendChild(style);
    },
    
    async nodeCreated(node) {
        const nodeType = node.comfyClass;
        
        // Setup logic based on node type
        switch(nodeType) {
            case "EmptyLatent":
                setupEmptyLatentLogic(node);
                break;
            case "ImageTransform":
                setupImageTransformLogic(node);
                break;
            case "MaskTransform":
                setupMaskTransformLogic(node);
                break;
            case "DepthDOF":
                setupDOFControls(node);
                setupNodeRemovalHandler(node, "dof");
                break;
            case "CameraDepthDOF":
                setupCamDOFControls(node);
                setupNodeRemovalHandler(node, "cdof");
                break;
            case "Saturation":
                setupSaturationControls(node);
                setupNodeRemovalHandler(node, "sat");
                break;
            case "Exposure":
                setupExposureControls(node);
                setupNodeRemovalHandler(node, "exp");
                break;
            case "Contrast":
                setupContrastControls(node);
                setupNodeRemovalHandler(node, "con");
                break;
            case "Brightness":
                setupBrightnessControls(node);
                setupNodeRemovalHandler(node, "bri");
                break;
            case "Temperature":
                setupTemperatureControls(node);
                setupNodeRemovalHandler(node, "tem");
                break;
            case "Hue":
                setupHueControls(node);
                setupNodeRemovalHandler(node, "hue");
                break;
            case "MaskFilter":
                setupMaskFilterControls(node);
                setupNodeRemovalHandler(node, "mfl");
                break;
            case "HighlightShadow":
                setupHighlightShadowControls(node);
                setupNodeRemovalHandler(node, "has");
                break;
            case "MaskProcessor":
                setupMaskProcessorControls(node);
                setupNodeRemovalHandler(node, "mpr");
                break;
            case "ImageTranslation":
                setupImageTranslationControls(node);
                setupNodeRemovalHandler(node, "itr");
                break;
            case "ImageRotation":
                setupImageRotationControls(node);
                setupNodeRemovalHandler(node, "iro");
                break;
            case "ImageZoom":
                setupImageZoomControls(node);
                setupNodeRemovalHandler(node, "izo");
                break;
            case "ImageResize":
                setupImageResizeControls(node);
                setupNodeRemovalHandler(node, "ire");
                break;
            case "MaskTranslation":
                setupMaskTranslationControls(node);
                setupNodeRemovalHandler(node, "mtr");
                break;
            case "MaskRotation":
                setupMaskRotationControls(node);
                setupNodeRemovalHandler(node, "mro");
                break;
            case "MaskZoom":
                setupMaskZoomControls(node);
                setupNodeRemovalHandler(node, "mzo");
                break;
            case "MaskResize":
                setupMaskResizeControls(node);
                setupNodeRemovalHandler(node, "mre");
                break;
            case "Dither":
                setupDitheringControls(node);
                setupNodeRemovalHandler(node, "dit");
                break;
        }
    }
});