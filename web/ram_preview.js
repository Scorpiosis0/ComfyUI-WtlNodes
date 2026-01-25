import { app } from "../../scripts/app.js";

// Session-only storage
class RAMImageStorage {
    constructor() {
        this.memoryCache = new Map();
    }

    saveImage(nodeId, base64Data) {
        this.memoryCache.set(nodeId, base64Data);
    }

    getImage(nodeId) {
        return this.memoryCache.get(nodeId) || null;
    }
}

const imageStorage = new RAMImageStorage();
const ramPreviewNodes = new Map();
let needsRefresh = false;

app.registerExtension({
    name: "RAMPreview.ImageDisplay",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RAMPreviewImage" ||
            nodeData.name === "Saturation" ||
            nodeData.name === "DepthDOF" ||
            nodeData.name === "Exposure" ||
            nodeData.name === "Contrast" ||
            nodeData.name === "Brightness" ||
            nodeData.name === "Temperature" ||
            nodeData.name === "Hue" ||
            nodeData.name === "MaskFilter" ||
            nodeData.name === "HighlightShadow" ||
            nodeData.name === "MaskProcessor" ||
            nodeData.name === "ImageZoom" ||
            nodeData.name === "ImageTranslation" ||
            nodeData.name === "ImageRotation" ||
            nodeData.name === "ImageResize" ||
            nodeData.name === "MaskZoom" ||
            nodeData.name === "MaskTranslation" ||
            nodeData.name === "MaskRotation" ||
            nodeData.name === "MaskResize" ||
            nodeData.name === "CameraDepthDOF" ||
            nodeData.name === "Dither"
        ) {
            
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                if (message?.ram_preview) {
                    this.imgs = [];
                    
                    message.ram_preview.forEach((base64Data) => {
                        const img = new Image();
                        img.src = `data:image/png;base64,${base64Data}`;
                        this.imgs.push(img);
                    });
                    
                    if (nodeData.name === "RAMPreviewImage") {
                        imageStorage.saveImage(this.id.toString(), message.ram_preview);
                        ramPreviewNodes.set(this.id, this);
                    }
                    
                    app.graph.setDirtyCanvas(true, true);
                }
            };

            // ONLY for RAMPreviewImage
            if (nodeData.name === "RAMPreviewImage") {
                const onAdded = nodeType.prototype.onAdded;
                nodeType.prototype.onAdded = function() {
                    if (onAdded) {
                        onAdded.apply(this, arguments);
                    }
                    ramPreviewNodes.set(this.id, this);
                };

                const onRemoved = nodeType.prototype.onRemoved;
                nodeType.prototype.onRemoved = function() {
                    ramPreviewNodes.delete(this.id);
                    if (onRemoved) {
                        onRemoved.apply(this, arguments);
                    }
                };

                // Trigger refresh when node is configured (undo/redo)
                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function(info) {
                    if (onConfigure) {
                        onConfigure.apply(this, arguments);
                    }
                    // Mark that we need to refresh images
                    needsRefresh = true;
                };
            }
        }
    }
});

// Only refresh when needed (after undo/redo)
setInterval(() => {
    if (needsRefresh) {
        for (const [nodeId, node] of ramPreviewNodes) {
            const cachedData = imageStorage.getImage(nodeId.toString());
            
            if (cachedData && Array.isArray(cachedData)) {
                node.imgs = [];
                cachedData.forEach((base64Data) => {
                    const img = new Image();
                    img.src = `data:image/png;base64,${base64Data}`;
                    node.imgs.push(img);
                });
            }
        }
        // Only dirty canvas once for all refreshes
        app.graph.setDirtyCanvas(true, true);
        needsRefresh = false;
    }
}, 100); // Check every 100ms but only refresh if needed