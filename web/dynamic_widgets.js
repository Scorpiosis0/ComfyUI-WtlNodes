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
    
    // Function to send updated parameters to Python
    const sendParams = () => {
        if (!focusDepthWidget || !focusRangeWidget || !edgeFixWidget || !hardFocusRangeWidget) return;
        
        fetch('/tgsz_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                node_id: node.id,
                node_type: 'dof',
                focus_depth: focusDepthWidget.value,
                focus_range: focusRangeWidget.value,
                edge_fix: edgeFixWidget.value,
                hard_focus_range : hardFocusRangeWidget.value
            })
        });
    };
    
    // Watch for slider changes and send updated params
    if (focusDepthWidget) {
        const origCallback = focusDepthWidget.callback;
        focusDepthWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }
    
    if (focusRangeWidget) {
        const origCallback = focusRangeWidget.callback;
        focusRangeWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }

    if (edgeFixWidget) {
        const origCallback = edgeFixWidget.callback;
        edgeFixWidget.callback = function(value) {
            sendParams();
            if (origCallback) origCallback.call(this, value);
        };
    }

    if (hardFocusRangeWidget) {
        const origCallback = hardFocusRangeWidget.callback;
        hardFocusRangeWidget.callback = function(value) {
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

function setupImageTransformLogic(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    
    if (!resizeByWidget) return;
    
    const updateVisibility = () => {
        const resizeBy = resizeByWidget.value;
        
        toggleWidget(node, widthWidget, resizeBy);
        toggleWidget(node, heightWidget, resizeBy);
        toggleWidget(node, multiplierWidget, !resizeBy);
        
        node.setDirtyCanvas(true);
    };
    
    // Initial setup
    updateVisibility();
    
    // Watch for changes using property descriptor
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
}

function setupMaskTransformLogic(node) {
    const resizeByWidget = findWidgetByName(node, "resize_by");
    const widthWidget = findWidgetByName(node, "width");
    const heightWidget = findWidgetByName(node, "height");
    const multiplierWidget = findWidgetByName(node, "multiplier");
    
    if (!resizeByWidget) return;
    
    const updateVisibility = () => {
        const resizeBy = resizeByWidget.value;
        
        toggleWidget(node, widthWidget, resizeBy);
        toggleWidget(node, heightWidget, resizeBy);
        toggleWidget(node, multiplierWidget, !resizeBy);
        
        node.setDirtyCanvas(true);
    };
    
    // Initial setup
    updateVisibility();
    
    // Watch for changes using property descriptor
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
            case "AdvancedEmptyLatent":
                setupEmptyLatentLogic(node);
                break;
            case "ImageTransformNode":
                setupImageTransformLogic(node);
                break;
            case "MaskTransformNode":
                setupMaskTransformLogic(node);
                break;
            case "DepthDOFNode":
                setupDOFControls(node);
                setupNodeRemovalHandler(node, "dof");
                break;
            case "saturationNode":
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
        }
    }
});