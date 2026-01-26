import { app } from "../../scripts/app.js";

// Session-only storage for compare images
class RAMCompareStorage {
    constructor() {
        this.memoryCache = new Map();
    }

    saveImages(nodeId, imagesBase64) {
        this.memoryCache.set(nodeId, imagesBase64);
    }

    getImages(nodeId) {
        return this.memoryCache.get(nodeId) || null;
    }
}

const compareStorage = new RAMCompareStorage();
const compareNodes = new Map();
let needsRefresh = false;

app.registerExtension({
    name: "RAMCompare.ImageDisplay",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RAMImageCompare") {
            
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                // Collect images from ram_preview messages
                if (message?.ram_preview) {
                    // Use compareImages instead of imgs to avoid default preview
                    this.compareImages = [];
                    this.compareImagesData = [];
                    
                    message.ram_preview.forEach((base64Data) => {
                        if (base64Data) {
                            const img = new Image();
                            img.src = `data:image/png;base64,${base64Data}`;
                            this.compareImages.push(img);
                            this.compareImagesData.push(base64Data);
                        }
                    });
                    
                    // Save to storage
                    compareStorage.saveImages(this.id.toString(), this.compareImagesData);
                    compareNodes.set(this.id, this);
                    
                    // Clear imgs to prevent default preview
                    this.imgs = [];
                    
                    app.graph.setDirtyCanvas(true, true);
                }
            };

            const onAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function() {
                if (onAdded) {
                    onAdded.apply(this, arguments);
                }
                this.sliderPos = null;
                this.isPointerOver = false;
                this.isPointerDown = false;
                this.compareImages = [];
                this.compareImagesData = [];
                
                // Get compare mode from widget
                this.updateCompareMode();
                
                compareNodes.set(this.id, this);
            };

            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                compareNodes.delete(this.id);
                if (onRemoved) {
                    onRemoved.apply(this, arguments);
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                
                // Update compare mode from widget
                this.updateCompareMode();
                
                needsRefresh = true;
            };

            // Helper to get compare mode from widget
            nodeType.prototype.updateCompareMode = function() {
                const widget = this.widgets?.find(w => w.name === "compare_mode");
                if (widget) {
                    this.compareMode = widget.value;
                } else {
                    this.compareMode = "slide";
                }
            };

            // Helper to update pointer state (like Rgthree)
            nodeType.prototype.setIsPointerDown = function(down) {
                const newIsDown = down && !!app.canvas.pointer_is_down;
                if (this.isPointerDown !== newIsDown) {
                    this.isPointerDown = newIsDown;
                    app.graph.setDirtyCanvas(true, false);
                }
                
                if (this.isPointerDown) {
                    requestAnimationFrame(() => {
                        this.setIsPointerDown(down);
                    });
                }
            };

            // Watch for widget changes
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function(name, value) {
                if (onWidgetChanged) {
                    onWidgetChanged.apply(this, arguments);
                }
                
                if (name === "compare_mode") {
                    this.compareMode = value;
                    app.graph.setDirtyCanvas(true, false);
                }
            };

            // Mouse interaction
            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function(e, localPos, canvas) {
                const result = onMouseMove ? onMouseMove.apply(this, arguments) : undefined;
                
                this.updateCompareMode(); // Update mode in case widget changed
                
                if (this.compareMode === "slide") {
                    this.sliderPos = localPos[0];
                    app.graph.setDirtyCanvas(true, false);
                }
                
                return result;
            };

            const onMouseEnter = nodeType.prototype.onMouseEnter;
            nodeType.prototype.onMouseEnter = function(e) {
                const result = onMouseEnter ? onMouseEnter.apply(this, arguments) : undefined;
                this.isPointerOver = true;
                
                // Update pointer down state based on canvas state
                this.setIsPointerDown(!!app.canvas.pointer_is_down);
                
                return result;
            };

            const onMouseLeave = nodeType.prototype.onMouseLeave;
            nodeType.prototype.onMouseLeave = function(e) {
                const result = onMouseLeave ? onMouseLeave.apply(this, arguments) : undefined;
                this.isPointerOver = false;
                this.sliderPos = null;
                
                // Stop pointer tracking
                this.setIsPointerDown(false);
                
                app.graph.setDirtyCanvas(true, false);
                return result;
            };

            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
                const result = onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
                
                this.updateCompareMode(); // Update mode
                
                if (this.compareMode === "click") {
                    this.setIsPointerDown(true);
                }
                
                return result;
            };

            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function(e, localPos, canvas) {
                const result = onMouseUp ? onMouseUp.apply(this, arguments) : undefined;
                
                // Stop tracking pointer
                this.setIsPointerDown(false);
                
                return result;
            };

            // Override default drawing completely
            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                if (onDrawBackground) {
                    onDrawBackground.apply(this, arguments);
                }
            };

            // Custom drawing for comparison
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                // Don't call original to prevent default image preview
                
                if (!this.compareImages || this.compareImages.length < 2) return;

                const [nodeWidth, nodeHeight] = this.size;
                const imgA = this.compareImages[0];
                const imgB = this.compareImages[1];
                
                if (!imgA || !imgA.naturalWidth || !imgA.naturalHeight) return;
                if (!imgB || !imgB.naturalWidth || !imgB.naturalHeight) return;

                this.updateCompareMode(); // Update mode before drawing

                // Calculate top padding (space for widgets and inputs)
                const topPadding = 80; // Space for dropdown and inputs
                const sidePadding = 10;
                const bottomPadding = 10;

                // Calculate dimensions for EACH image independently
                const calculateImageDimensions = (img) => {
                    const aspect = img.naturalWidth / img.naturalHeight;
                    const availableHeight = nodeHeight - topPadding - bottomPadding;
                    const availableWidth = nodeWidth - sidePadding * 2;
                    
                    let width, height;
                    
                    if (availableWidth / availableHeight > aspect) {
                        height = availableHeight;
                        width = height * aspect;
                    } else {
                        width = availableWidth;
                        height = width / aspect;
                    }
                    
                    const offsetX = (nodeWidth - width) / 2;
                    const offsetY = topPadding + (availableHeight - height) / 2;
                    
                    return { width, height, offsetX, offsetY };
                };

                const dimsA = calculateImageDimensions(imgA);
                const dimsB = calculateImageDimensions(imgB);

                // Draw image A (base layer)
                ctx.save();
                ctx.drawImage(imgA, dimsA.offsetX, dimsA.offsetY, dimsA.width, dimsA.height);
                ctx.restore();

                // Draw image B (overlay with clipping or full)
                let showImageB = false;
                let clipX = nodeWidth / 2;

                if (this.compareMode === "click") {
                    // Only show B while mouse is HELD DOWN
                    showImageB = this.isPointerDown;
                } else {
                    // Slide mode
                    showImageB = this.isPointerOver && this.sliderPos !== null;
                    if (showImageB) {
                        clipX = this.sliderPos;
                    }
                }

                if (showImageB && this.compareMode === "slide") {
                    // Slide mode: reveal from LEFT to RIGHT
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(0, topPadding, clipX, nodeHeight - topPadding);
                    ctx.clip();
                    ctx.drawImage(imgB, dimsB.offsetX, dimsB.offsetY, dimsB.width, dimsB.height);
                    ctx.restore();

                    // Draw slider line
                    ctx.save();
                    ctx.beginPath();
                    ctx.moveTo(clipX, topPadding);
                    ctx.lineTo(clipX, nodeHeight);
                    ctx.globalCompositeOperation = "difference";
                    ctx.strokeStyle = "rgba(255, 255, 255, 1)";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.restore();
                } else if (showImageB && this.compareMode === "click") {
                    // Click mode: show full image B while holding
                    ctx.save();
                    ctx.drawImage(imgB, dimsB.offsetX, dimsB.offsetY, dimsB.width, dimsB.height);
                    ctx.restore();
                }
            };
        }
    }
});

// Refresh after undo/redo
setInterval(() => {
    if (needsRefresh) {
        for (const [nodeId, node] of compareNodes) {
            const cachedData = compareStorage.getImages(nodeId.toString());
            
            if (cachedData && Array.isArray(cachedData)) {
                node.compareImages = [];
                node.compareImagesData = cachedData;
                cachedData.forEach((base64Data) => {
                    const img = new Image();
                    img.src = `data:image/png;base64,${base64Data}`;
                    node.compareImages.push(img);
                });
                // Keep imgs empty to prevent default preview
                node.imgs = [];
            }
        }

        app.graph.setDirtyCanvas(true, true);
        needsRefresh = false;
    }
}, 100);