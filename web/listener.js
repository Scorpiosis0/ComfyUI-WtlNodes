/*import { app } from "../../scripts/app.js";

console.log('[ComfyUI Monitor] Starting event monitoring...');

// Monitor LGraph events
if (app.graph) {
    console.log('[ComfyUI Monitor] Graph found, attaching listeners...');
    
    // Graph change events
    const originalChange = app.graph.change;
    app.graph.change = function() {
        console.log('[ComfyUI Monitor] 🔄 graph.change()');
        if (originalChange) {
            return originalChange.apply(this, arguments);
        }
    };

    // Canvas dirty events
    const originalSetDirtyCanvas = app.graph.setDirtyCanvas;
    app.graph.setDirtyCanvas = function(fg, bg) {
        console.log('[ComfyUI Monitor] 🎨 setDirtyCanvas(fg:', fg, 'bg:', bg, ')');
        if (originalSetDirtyCanvas) {
            return originalSetDirtyCanvas.apply(this, arguments);
        }
    };

    // Node added
    const originalOnNodeAdded = app.graph.onNodeAdded;
    app.graph.onNodeAdded = function(node) {
        console.log('[ComfyUI Monitor] ➕ Node added:', node.id, node.type);
        if (originalOnNodeAdded) {
            return originalOnNodeAdded.apply(this, arguments);
        }
    };

    // Node removed
    const originalOnNodeRemoved = app.graph.onNodeRemoved;
    app.graph.onNodeRemoved = function(node) {
        console.log('[ComfyUI Monitor] ➖ Node removed:', node.id, node.type);
        if (originalOnNodeRemoved) {
            return originalOnNodeRemoved.apply(this, arguments);
        }
    };
}

// Monitor LGraphCanvas (rendering) events
if (app.canvas) {
    console.log('[ComfyUI Monitor] Canvas found, attaching listeners...');
    
    // Draw events
    const originalDraw = app.canvas.draw;
    app.canvas.draw = function(force_canvas, force_bgcanvas) {
        console.log('[ComfyUI Monitor] 🖼️ canvas.draw(force_canvas:', force_canvas, 'force_bgcanvas:', force_bgcanvas, ')');
        if (originalDraw) {
            return originalDraw.apply(this, arguments);
        }
    };
}

// Monitor API events
if (app.api) {
    console.log('[ComfyUI Monitor] API found, attaching listeners...');
    
    const originalDispatchEvent = app.api.dispatchEvent;
    app.api.dispatchEvent = function(event) {
        console.log('[ComfyUI Monitor] 📡 API event:', event.type, event.detail ? event.detail : '');
        if (originalDispatchEvent) {
            return originalDispatchEvent.apply(this, arguments);
        }
    };
}

// Monitor keyboard events
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
        console.log('[ComfyUI Monitor] ⌨️ Keyboard:', e.key, 'ctrl/cmd:', e.ctrlKey || e.metaKey);
    }
});

// Monitor undo/redo specifically
let undoRedoCount = 0;
const checkUndoRedo = () => {
    if (app.graph && app.graph.list_of_graphcanvas) {
        app.graph.list_of_graphcanvas.forEach(canvas => {
            if (canvas.history_index !== undoRedoCount) {
                console.log('[ComfyUI Monitor] ↩️ UNDO/REDO detected! History index:', canvas.history_index);
                undoRedoCount = canvas.history_index;
            }
        });
    }
};
setInterval(checkUndoRedo, 100);

// Monitor extension registration
app.registerExtension({
    name: "ComfyUI.Monitor",
    
    async setup() {
        console.log('[ComfyUI Monitor] 🚀 Extension setup()');
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log('[ComfyUI Monitor] 📦 beforeRegisterNodeDef:', nodeData.name);
        
        // Monitor all node lifecycle events
        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            console.log('[ComfyUI Monitor] ✅ onExecuted for node:', this.id, this.type);
            if (originalOnExecuted) {
                return originalOnExecuted.apply(this, arguments);
            }
        };

        const originalOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function() {
            console.log('[ComfyUI Monitor] ➕ Node onAdded:', this.id, this.type);
            if (originalOnAdded) {
                return originalOnAdded.apply(this, arguments);
            }
        };

        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            console.log('[ComfyUI Monitor] ➖ Node onRemoved:', this.id, this.type);
            if (originalOnRemoved) {
                return originalOnRemoved.apply(this, arguments);
            }
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            console.log('[ComfyUI Monitor] ⚙️ Node onConfigure:', this.id, this.type);
            if (originalOnConfigure) {
                return originalOnConfigure.apply(this, arguments);
            }
        };

        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info) {
            console.log('[ComfyUI Monitor] 💾 Node onSerialize:', this.id, this.type);
            if (originalOnSerialize) {
                return originalOnSerialize.apply(this, arguments);
            }
        };

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            console.log('[ComfyUI Monitor] 🎨 Node onDrawForeground:', this.id, this.type);
            if (originalOnDrawForeground) {
                return originalOnDrawForeground.apply(this, arguments);
            }
        };

        const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            console.log('[ComfyUI Monitor] 🖼️ Node onDrawBackground:', this.id, this.type);
            if (originalOnDrawBackground) {
                return originalOnDrawBackground.apply(this, arguments);
            }
        };
    },
    
    async nodeCreated(node) {
        console.log('[ComfyUI Monitor] 🆕 nodeCreated:', node.id, node.type);
    },
    
    async loadedGraphNode(node) {
        console.log('[ComfyUI Monitor] 📂 loadedGraphNode:', node.id, node.type);
    }
});

console.log('[ComfyUI Monitor] Monitoring active! Press Ctrl+Z to see what happens.');
*/