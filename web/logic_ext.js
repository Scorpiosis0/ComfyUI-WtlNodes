import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
// WtlNodes / Logic — front-end extension
//
// Binary math nodes (Add, Subtract, Multiply, Divide):
//   Each has an `x` socket (INT,FLOAT) plus two widgets: int_value and
//   float_value.  JS shows whichever matches the connected type and hides the
//   other, using the same toggleWidget pattern as dynamic_widgets.js.
//   Output type follows the connected type.
//
// Square / Square Root:
//   Single INT,FLOAT socket, type propagates to output.
//
// Text Append:
//   Starts with text_1 + text_2 (forceInput, always pure sockets).
//   Connecting the last slot auto-adds the next one.
// ─────────────────────────────────────────────────────────────────────────────

const DUAL_TYPE    = "INT,FLOAT";
const MATH_NODES   = ["WtlAdd", "WtlSubtract", "WtlMultiply", "WtlDivide", "WtlSquare"];
const BINARY_NODES = new Set(["WtlAdd", "WtlSubtract", "WtlMultiply", "WtlDivide"]);

// ─────────────────────────────────────────────────────────────────────────────
// WtlCast: sync output type to the cast_to_int boolean toggle
// ─────────────────────────────────────────────────────────────────────────────
function syncCastOutput(node) {
    const w = node.widgets?.find(w => w.name === "cast_to_int");
    if (!w || !node.outputs?.[0]) return;
    const t = w.value ? "INT" : "FLOAT";
    node.outputs[0].type  = t;
    node.outputs[0].name  = t;
    node.outputs[0].label = t;
    app.graph.setDirtyCanvas(true, false);
}

// Same hidden tag and pattern as dynamic_widgets.js
const HIDDEN_TAG = "tgszhidden";

// ─────────────────────────────────────────────────────────────────────────────
// toggleWidget — identical logic to dynamic_widgets.js
//   force = true  → hide
//   force = false → show
// ─────────────────────────────────────────────────────────────────────────────
function toggleWidget(node, widget, force) {
    if (!widget) return;

    widget.options[HIDDEN_TAG] ??= (
        widget.options.origType        = widget.type,
        widget.options.origComputeSize = widget.computeSize,
        HIDDEN_TAG
    );
    const hide = force ?? (widget.type !== HIDDEN_TAG);

    widget.type          = hide ? widget.options[HIDDEN_TAG]    : widget.options.origType;
    widget.hidden        = hide ? true                          : undefined;
    widget.computeSize   = hide ? () => [0, -3.3]              : widget.options.origComputeSize;
    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, force));

    for (const el of ["inputEl", "input"])
        widget[el]?.classList?.toggle(HIDDEN_TAG, force);

    const height = hide ? node.size[1] : Math.max(node.computeSize()[1], node.size[1]);
    node.setSize([node.size[0], height]);
    widget.computedHeight = hide ? 0 : undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — resolve the actual type of a connected link
// ─────────────────────────────────────────────────────────────────────────────
function resolveLinkedType(node, slotIndex) {
    const link_id = node.inputs[slotIndex]?.link;
    if (link_id == null) return null;
    const link = app.graph.links[link_id];
    if (!link) return null;
    const srcNode = app.graph.getNodeById(link.origin_id);
    if (!srcNode) return null;
    return srcNode.outputs[link.origin_slot]?.type ?? null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — set all DUAL_TYPE inputs + output to a concrete type, or reset
// ─────────────────────────────────────────────────────────────────────────────
function applyType(node, type) {
    for (const inp of node.inputs ?? []) {
        if (inp.type === DUAL_TYPE || inp.type === "INT" || inp.type === "FLOAT") {
            inp.type = type;
        }
    }
    for (const out of node.outputs ?? []) {
        if (out.type === DUAL_TYPE || out.type === "INT" || out.type === "FLOAT") {
            out.type  = type;
            out.label = type;
            out.name  = type;
        }
    }
    app.graph.setDirtyCanvas(true, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — scan all inputs and propagate INT/FLOAT, or reset to dual
// ─────────────────────────────────────────────────────────────────────────────
function syncTypeFromConnections(node) {
    for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
        const t = resolveLinkedType(node, i);
        if (t === "INT" || t === "FLOAT") {
            applyType(node, t);
            return;
        }
    }
    applyType(node, DUAL_TYPE);
}

// ─────────────────────────────────────────────────────────────────────────────
// Show int_value or float_value based on what's connected to x (slot 0)
// ─────────────────────────────────────────────────────────────────────────────
function syncValueWidgets(node) {
    const intW   = node.widgets?.find(w => w.name === "int_value");
    const floatW = node.widgets?.find(w => w.name === "float_value");
    if (!intW || !floatW) return;

    const t = resolveLinkedType(node, 0);  // x is always slot 0
    if (t === "INT") {
        toggleWidget(node, intW,   false);  // show int
        toggleWidget(node, floatW, true);   // hide float
    } else {
        // FLOAT or nothing connected — show float
        toggleWidget(node, intW,   true);   // hide int
        toggleWidget(node, floatW, false);  // show float
    }
}

// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "WtlNodes.Logic",

    // ── Initial widget visibility ─────────────────────────────────────────────
    async nodeCreated(node) {
        // Binary math nodes: default state — show float_value, hide int_value
        if (BINARY_NODES.has(node.comfyClass)) {
            const intW = node.widgets?.find(w => w.name === "int_value");
            if (intW) toggleWidget(node, intW, true);
        }

        // Cast node: wire up cast_to_int toggle to update the output type live
        if (node.comfyClass === "WtlCast") {
            syncCastOutput(node);
            const castW = node.widgets?.find(w => w.name === "cast_to_int");
            if (castW) {
                let val = castW.value;
                Object.defineProperty(castW, "value", {
                    get() { return val; },
                    set(v) {
                        if (v !== val) { val = v; syncCastOutput(node); }
                    },
                });
            }
        }

    },

    async beforeRegisterNodeDef(nodeType, nodeData) {

        // ── Math nodes: type propagation + int/float widget swap ─────────────
        if (MATH_NODES.includes(nodeData.name)) {

            const origOnConnChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (origOnConnChange) origOnConnChange.apply(this, arguments);
                if (type !== 1 /* INPUT */) return;
                syncTypeFromConnections(this);
                if (BINARY_NODES.has(nodeData.name)) syncValueWidgets(this);
            };

            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
                setTimeout(() => {
                    syncTypeFromConnections(this);
                    if (BINARY_NODES.has(nodeData.name)) syncValueWidgets(this);
                }, 0);
            };
        }

        // ── Cast node: sync output type on workflow load ──────────────────────
        if (nodeData.name === "WtlCast") {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (origOnConfigure) origOnConfigure.apply(this, arguments);
                setTimeout(() => syncCastOutput(this), 0);
            };
        }

        // ── Text node: write executed value into the existing textarea ────────
        if (nodeData.name === "WtlText") {
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                const text = message?.text_display?.[0];
                if (text === undefined) return;
                const w = this.widgets?.find(w => w.name === "value");
                if (w) {
                    w.value = text;
                    app.graph.setDirtyCanvas(true, false);
                }
            };
        }

        // ── Text Append: dynamic slots ────────────────────────────────────────
        if (nodeData.name === "WtlTextAppend") {

            const origOnConnChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (origOnConnChange) origOnConnChange.apply(this, arguments);
                if (type !== 1 /* INPUT */) return;

                // If separator was just disconnected, revert it back to a widget.
                // ComfyUI's removeInput restores the widget automatically when the
                // input has a .widget property (set during "Convert to Input").
                const sepInp = this.inputs.find(i => i.name === "separator");
                if (sepInp && !sepInp.link && sepInp.widget) {
                    this.removeInput(this.inputs.indexOf(sepInp));
                    return;
                }

                const stackTrace = new Error().stack;
                if (stackTrace.includes("loadGraphData") || stackTrace.includes("configure")) return;

                if (this._textManaging) return;
                this._textManaging = true;

                try {
                    const textInps = this.inputs.filter(i => i.name.startsWith("text_"));

                    // Trim trailing unconnected text slots (keep at least text_1 and text_2)
                    while (textInps.length > 2 && !textInps[textInps.length - 1].link) {
                        const realIdx = this.inputs.indexOf(textInps.pop());
                        this.removeInput(realIdx);
                    }

                    // Add next slot if the last text slot is now connected
                    if (textInps[textInps.length - 1]?.link) {
                        this.addInput(`text_${textInps.length + 1}`, "STRING");

                        // Keep separator at the very end if it wound up in inputs
                        const sepInp = this.inputs.find(i => i.name === "separator");
                        if (sepInp) {
                            this.inputs.splice(this.inputs.indexOf(sepInp), 1);
                            this.inputs.push(sepInp);
                        }
                    }

                    // Re-label text slots sequentially
                    let n = 1;
                    for (const inp of this.inputs) {
                        if (inp.name.startsWith("text_")) inp.name = `text_${n++}`;
                    }
                } finally {
                    this._textManaging = false;
                }
            };
        }
    },
});
