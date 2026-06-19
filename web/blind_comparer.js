import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
// Blind Comparer — front-end extension
//
// Python sends two separate ram_preview frames per match (left, right).
// ComfyUI renders both stacked on the node canvas naturally.
// The DOM widget contains only text labels + two vote buttons — no thumbnails.
// ─────────────────────────────────────────────────────────────────────────────

const FONT = `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;

app.registerExtension({
    name: "WtlNodes.BlindComparer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "BlindComparer") return;

        // ── Dynamic input slots ───────────────────────────────────────────────
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (onConnectionsChange) onConnectionsChange.apply(this, arguments);

            if (type !== 1 /* INPUT */) return;

            const stackTrace = new Error().stack;
            if (stackTrace.includes("loadGraphData") || stackTrace.includes("configure")) return;

            // Trim trailing unconnected slots, keep minimum one
            while (this.inputs.length > 1 && !this.inputs[this.inputs.length - 1].link) {
                this.removeInput(this.inputs.length - 1);
            }

            // Always keep one empty slot at the end
            if (this.inputs[this.inputs.length - 1].link) {
                this.addInput(`image_${this.inputs.length + 1}`, "IMAGE");
            }

            // Re-label sequentially
            this.inputs.forEach((inp, i) => { inp.name = `image_${i + 1}`; });
        };

        // ── onExecuted ────────────────────────────────────────────────────────
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);

            // Both frames go into this.imgs — ComfyUI renders them on the canvas
            if (message?.ram_preview) {
                this.imgs = message.ram_preview.map(b64 => {
                    const img = new Image();
                    img.src = `data:image/png;base64,${b64}`;
                    return img;
                });
                app.graph.setDirtyCanvas(true, true);
            }

            if (message?.bracket_match) {
                this._updateMatchUI(message.bracket_match);
            }

            if (message?.bracket_champion) {
                this._showChampion(message.bracket_champion);
            }
        };

        // ── onNodeCreated ─────────────────────────────────────────────────────
        const origNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origNodeCreated) origNodeCreated.apply(this, arguments);
            _buildUI(this);
        };

        // ── skip on removal ───────────────────────────────────────────────────
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            fetch("/wtl_bracket_skip", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: this.id }),
            }).catch(() => {});
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
        };
    },
});

// ─────────────────────────────────────────────────────────────────────────────
function _buildUI(node) {

    const wrap = document.createElement("div");
    wrap.style.cssText = `
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding: 5px 8px 9px 8px;
        width: 100%;
        box-sizing: border-box;
    `;

    // ── match label ───────────────────────────────────────────────────────────
    const matchLabel = document.createElement("div");
    matchLabel.style.cssText = `
        color: #71717a;
        font: 500 10px ${FONT};
        text-align: center;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        min-height: 13px;
    `;
    wrap.appendChild(matchLabel);

    // ── vote buttons — text only, no thumbnails ───────────────────────────────
    const btnRow = document.createElement("div");
    btnRow.style.cssText = `display: flex; gap: 5px; width: 100%;`;

    const makeBtn = (side, arrow, label, hov, act) => {
        const btn = document.createElement("button");
        btn.style.cssText = `
            flex: 1;
            padding: 7px 4px;
            background: #52525b;
            border: 2px solid #3f3f46;
            border-radius: 2px;
            color: #ffffff;
            font: 600 12px ${FONT};
            cursor: pointer;
            transition: background 0.12s;
        `;
        btn.textContent = `${arrow} ${label}`;

        btn.addEventListener("mouseenter", () => { btn.style.background = hov; });
        btn.addEventListener("mouseleave", () => { btn.style.background = "#52525b"; });
        btn.addEventListener("mousedown",  () => { btn.style.background = act; });
        btn.addEventListener("mouseup",    () => { btn.style.background = hov; });
        btn.addEventListener("click", e => {
            e.stopPropagation();
            fetch("/wtl_bracket_vote", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: node.id, action: side }),
            }).catch(err => console.error("[BlindComparer] vote error:", err));
        });

        return btn;
    };

    const leftBtn  = makeBtn("left",  "◀", "Left",  "#86a089", "#5a6f5c");
    const rightBtn = makeBtn("right", "▶", "Right", "#8b9fc4", "#5d7196");
    btnRow.appendChild(leftBtn);
    btnRow.appendChild(rightBtn);
    wrap.appendChild(btnRow);

    // ── champion line ─────────────────────────────────────────────────────────
    const champLine = document.createElement("div");
    champLine.style.cssText = `
        display: none;
        background: #3b4a3c;
        border: 2px solid #5a6f5c;
        border-radius: 2px;
        color: #a8d5ab;
        font: 600 12px ${FONT};
        text-align: center;
        padding: 6px 8px;
        letter-spacing: 0.02em;
    `;
    wrap.appendChild(champLine);

    // ── DOM widget ────────────────────────────────────────────────────────────
    const domWidget = node.addDOMWidget("blind_comparer_ui", "custom", wrap, {
        serialize: false,
        hideOnZoom: false,
        getValue() { return null; },
        setValue() {},
    });
    // matchLabel(18) + btnRow(~36) + gaps+pad(~20)
    domWidget.computeSize = w => [w, 79];

    // ── node methods ──────────────────────────────────────────────────────────

    node._updateMatchUI = function (match) {
        champLine.style.display = "none";
        leftBtn.style.display   = "";
        rightBtn.style.display  = "";
        domWidget.computeSize   = w => [w, 79];

        matchLabel.textContent = `Round ${match.round + 1}`;
    };

    node._showChampion = function (label) {
        champLine.textContent   = `🏆  ${label} is the preferred image`;
        champLine.style.display = "block";
        leftBtn.style.display   = "none";
        rightBtn.style.display  = "none";
        matchLabel.textContent  = "";

        domWidget.computeSize = w => [w, 48];
        node.setSize([node.size[0], node.computeSize()[1]]);
        app.graph.setDirtyCanvas(true, true);
    };
}