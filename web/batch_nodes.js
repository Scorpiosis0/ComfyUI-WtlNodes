import { app } from "../../scripts/app.js";

// ── Dynamic slot management ────────────────────────────────────────────────

function setupBatchNode(nodeType, prefix, slotType) {
    const origOnConnChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
        if (origOnConnChange) origOnConnChange.apply(this, arguments);
        if (type !== 1 /* INPUT */) return;

        const stackTrace = new Error().stack;
        if (stackTrace.includes("loadGraphData") || stackTrace.includes("configure")) return;

        if (this._batchManaging) return;
        this._batchManaging = true;
        try {
            const slotInps = this.inputs.filter(i => i.name.startsWith(prefix));

            while (slotInps.length > 2 && !slotInps[slotInps.length - 1].link) {
                const realIdx = this.inputs.indexOf(slotInps.pop());
                this.removeInput(realIdx);
            }
            if (slotInps[slotInps.length - 1]?.link) {
                this.addInput(`${prefix}${slotInps.length + 1}`, slotType);
            }
            let n = 1;
            for (const inp of this.inputs) {
                if (inp.name.startsWith(prefix)) inp.name = `${prefix}${n++}`;
            }
        } finally {
            this._batchManaging = false;
        }
    };
}

// ── ZIP helpers ────────────────────────────────────────────────────────────

function crc32(data) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < data.length; i++) {
        crc ^= data[i];
        for (let j = 0; j < 8; j++) {
            crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
        }
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
}

function createZip(files) {
    const parts = [];
    const centralDir = [];
    let offset = 0;

    for (const file of files) {
        const nameBytes = new TextEncoder().encode(file.name);
        const crc = crc32(file.data);

        const lh = new DataView(new ArrayBuffer(30 + nameBytes.length));
        lh.setUint32(0,  0x04034b50, true); // local file header sig
        lh.setUint16(4,  20,         true); // version needed
        lh.setUint16(6,  0,          true); // general purpose flag
        lh.setUint16(8,  0,          true); // compression (stored)
        lh.setUint32(10, 0,          true); // last mod time/date
        lh.setUint32(14, crc,        true); // crc-32
        lh.setUint32(18, file.data.length, true); // compressed size
        lh.setUint32(22, file.data.length, true); // uncompressed size
        lh.setUint16(26, nameBytes.length, true); // file name length
        lh.setUint16(28, 0,          true); // extra field length
        new Uint8Array(lh.buffer).set(nameBytes, 30);

        const cd = new DataView(new ArrayBuffer(46 + nameBytes.length));
        cd.setUint32(0,  0x02014b50, true); // central dir sig
        cd.setUint16(4,  20,         true); // version made by
        cd.setUint16(6,  20,         true); // version needed
        cd.setUint16(8,  0,          true);
        cd.setUint16(10, 0,          true);
        cd.setUint32(12, 0,          true);
        cd.setUint32(16, crc,        true);
        cd.setUint32(20, file.data.length, true);
        cd.setUint32(24, file.data.length, true);
        cd.setUint16(28, nameBytes.length, true);
        cd.setUint16(30, 0,          true); // extra length
        cd.setUint16(32, 0,          true); // comment length
        cd.setUint16(34, 0,          true); // disk start
        cd.setUint16(36, 0,          true); // internal attrs
        cd.setUint32(38, 0,          true); // external attrs
        cd.setUint32(42, offset,     true); // local header offset
        new Uint8Array(cd.buffer).set(nameBytes, 46);

        parts.push(new Uint8Array(lh.buffer));
        parts.push(file.data);
        centralDir.push(new Uint8Array(cd.buffer));
        offset += 30 + nameBytes.length + file.data.length;
    }

    const cdSize = centralDir.reduce((s, c) => s + c.length, 0);
    const eocd = new DataView(new ArrayBuffer(22));
    eocd.setUint32(0,  0x06054b50,   true); // end of central dir sig
    eocd.setUint16(4,  0,            true);
    eocd.setUint16(6,  0,            true);
    eocd.setUint16(8,  files.length, true);
    eocd.setUint16(10, files.length, true);
    eocd.setUint32(12, cdSize,       true);
    eocd.setUint32(16, offset,       true);
    eocd.setUint16(20, 0,            true);

    const allParts = [...parts, ...centralDir, new Uint8Array(eocd.buffer)];
    const total = allParts.reduce((s, p) => s + p.length, 0);
    const result = new Uint8Array(total);
    let pos = 0;
    for (const p of allParts) { result.set(p, pos); pos += p.length; }
    return result;
}

function b64ToBytes(b64) {
    const chars = atob(b64);
    const arr = new Uint8Array(chars.length);
    for (let i = 0; i < chars.length; i++) arr[i] = chars.charCodeAt(i);
    return arr;
}

// ── Image Batch buttons ────────────────────────────────────────────────────

function setupImageBatchButtons(nodeType) {
    // Capture frame data alongside the existing ram_preview.js hook
    const origOnExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        if (origOnExecuted) origOnExecuted.apply(this, arguments);
        if (message?.ram_preview) {
            this.batchFramesData = message.ram_preview;
        }
    };



    // Save ZIP: all frames packed into one archive
    nodeType.prototype.saveBatchZip = function () {
        if (!this.batchFramesData?.length) return;
        const files = this.batchFramesData.map((b64, i) => ({
            name: `image_${i + 1}.png`,
            data: b64ToBytes(b64),
        }));
        const zip = createZip(files);
        const blob = new Blob([zip], { type: "application/zip" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `image_batch_${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);

        const container = document.createElement("div");
        container.style.cssText = "display:flex;flex-direction:column;gap:4px;padding:4px 8px 8px 8px;width:100%;";

        // Save ZIP (full width, purple)
        const zipBtn = document.createElement("button");
        zipBtn.textContent = "📦 Save as ZIP";
        zipBtn.className = "wtl-batch-save";
        zipBtn.style.width = "100%";
        zipBtn.addEventListener("click", e => { e.stopPropagation(); this.saveBatchZip(); });

        container.appendChild(zipBtn);

        const buttonHeight = 28;
        const totalHeight = 12 + buttonHeight; // padding + 1 row

        const widget = this.addDOMWidget("batch_buttons", "custom", container, {
            serialize: false,
            hideOnZoom: false,
            getValue() { return null; },
            setValue() {},
        });
        widget.computeSize = function (width) { return [width, totalHeight]; };
    };
}

// ── Extension ──────────────────────────────────────────────────────────────

app.registerExtension({
    name: "WtlNodes.BatchNodes",

    async init() {
        const style = document.createElement("style");
        style.textContent = `
            .wtl-batch-save {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                padding: 4px 8px !important;
                border-radius: 2px !important;
                cursor: pointer !important;
                transition: background-color 0.15s ease, border-color 0.15s ease !important;
                outline: none !important;
                letter-spacing: 0.01em !important;
                background-color: #52525b !important;
                border: 2px solid #3f3f46 !important;
                color: #ffffff !important;
            }
            .wtl-batch-save:hover {
                background-color: #8b5cf6 !important;
                border-color: #7c3aed !important;
            }
            .wtl-batch-save:active {
                background-color: #7c3aed !important;
                border-color: #6d28d9 !important;
            }
        `;
        document.head.appendChild(style);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "WtlImageBatch" || nodeData.name === "WtlImageCombiner") {
            setupBatchNode(nodeType, "image_", "IMAGE");
        }
        if (nodeData.name === "WtlImageCombiner") {
            setupImageBatchButtons(nodeType);
        }
        if (nodeData.name === "WtlMaskBatch" || nodeData.name === "WtlMaskCombiner") {
            setupBatchNode(nodeType, "mask_", "MASK");
        }
    },
});
