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
    
    async init() {
        const style = document.createElement("style");
        style.textContent = `
            /* RAM Compare button styles */
            .ram-compare-save,
            .ram-compare-copy {
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
            
            /* Save buttons */
            .ram-compare-save:hover {
                background-color: #8b5cf6 !important;
                border-color: #7c3aed !important;
            }
            
            .ram-compare-save:active {
                background-color: #7c3aed !important;
                border-color: #6d28d9 !important;
            }
            
            /* Copy buttons - Square */
            .ram-compare-copy {
                width: 28px !important;
                height: 28px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            
            .ram-compare-copy:hover {
                background-color: #facc15 !important;
                border-color: #eab308 !important;
            }
            
            .ram-compare-copy:active {
                background-color: #eab308 !important;
                border-color: #ca8a04 !important;
            }
        `;
        document.head.appendChild(style);
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RAMImageCompare") {
            
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                if (message?.ram_preview) {
                    this.compareImages = [];
                    this.compareImagesData = [];
                    
                    message.ram_preview.forEach((base64Data) => {
                        if (base64Data) {
                            const img = new Image();
                            img.onload = () => {
                                app.graph.setDirtyCanvas(true, false);
                            };
                            img.src = `data:image/png;base64,${base64Data}`;
                            this.compareImages.push(img);
                            this.compareImagesData.push(base64Data);
                        }
                    });
                    
                    compareStorage.saveImages(this.id.toString(), this.compareImagesData);
                    compareNodes.set(this.id, this);
                    
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
                
                this.updateCompareMode();
                
                needsRefresh = true;
            };

            nodeType.prototype.updateCompareMode = function() {
                const widget = this.widgets?.find(w => w.name === "compare_mode");
                if (widget) {
                    this.compareMode = widget.value;
                } else {
                    this.compareMode = "slide";
                }
            };

            nodeType.prototype.saveImage = function(imageIndex) {
                if (!this.compareImagesData || !this.compareImagesData[imageIndex]) {
                    console.error("No image data available");
                    return;
                }
                
                const base64Data = this.compareImagesData[imageIndex];
                const imageName = imageIndex === 0 ? "A" : "B";
                
                const byteCharacters = atob(base64Data);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: 'image/png' });
                
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `compare_image_${imageName}_${Date.now()}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            };

            nodeType.prototype.copyImage = function(imageIndex) {
                if (!this.compareImagesData || !this.compareImagesData[imageIndex]) {
                    console.error("No image data available");
                    return;
                }
                
                const base64Data = this.compareImagesData[imageIndex];
                
                const byteCharacters = atob(base64Data);
                const byteNumbers = new Array(byteCharacters.length);
                for (let i = 0; i < byteCharacters.length; i++) {
                    byteNumbers[i] = byteCharacters.charCodeAt(i);
                }
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], { type: 'image/png' });
                
                navigator.clipboard.write([
                    new ClipboardItem({ 'image/png': blob })
                ]).then(() => {
                    console.log('Image copied to clipboard');
                }).catch(err => {
                    console.error('Failed to copy image:', err);
                });
            };

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

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Create button rows container
                const buttonsContainer = document.createElement("div");
                buttonsContainer.style.cssText = `
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                    padding: 4px 8px 8px 8px;
                    width: 100%;
                `;
                
                // Row A
                const rowA = document.createElement("div");
                rowA.style.cssText = `
                    display: flex;
                    gap: 4px;
                    width: 100%;
                `;
                
                const saveButtonA = document.createElement("button");
                saveButtonA.textContent = '💾 Save Image A';
                saveButtonA.className = 'ram-compare-save';
                saveButtonA.style.flex = '1';
                saveButtonA.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.saveImage(0);
                });
                
                const copyButtonA = document.createElement("button");
                copyButtonA.textContent = '📋';
                copyButtonA.className = 'ram-compare-copy';
                copyButtonA.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.copyImage(0);
                });
                
                rowA.appendChild(saveButtonA);
                rowA.appendChild(copyButtonA);
                
                // Row B
                const rowB = document.createElement("div");
                rowB.style.cssText = `
                    display: flex;
                    gap: 4px;
                    width: 100%;
                `;
                
                const saveButtonB = document.createElement("button");
                saveButtonB.textContent = '💾 Save Image B';
                saveButtonB.className = 'ram-compare-save';
                saveButtonB.style.flex = '1';
                saveButtonB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.saveImage(1);
                });
                
                const copyButtonB = document.createElement("button");
                copyButtonB.textContent = '📋';
                copyButtonB.className = 'ram-compare-copy';
                copyButtonB.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.copyImage(1);
                });
                
                rowB.appendChild(saveButtonB);
                rowB.appendChild(copyButtonB);
                
                buttonsContainer.appendChild(rowA);
                buttonsContainer.appendChild(rowB);
                
                // Calculate button container height
                const containerPadding = 12;
                const buttonHeight = 28;
                const gapHeight = 4;
                const containerHeight = containerPadding + (2 * buttonHeight) + gapHeight;
                
                const buttonWidget = this.addDOMWidget("buttons", "custom", buttonsContainer, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() { return null; },
                    setValue(v) {}
                });
                
                buttonWidget.computeSize = function(width) {
                    return [width, containerHeight];
                };
                
                const minWidth = 300;
                const minHeight = 200;
                
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    size[0] = Math.max(size[0], minWidth);
                    size[1] = Math.max(size[1], minHeight);
                    
                    if (originalOnResize) {
                        return originalOnResize.apply(this, arguments);
                    }
                };
                
                if (this.size[0] < minWidth || this.size[1] < minHeight) {
                    this.size = [
                        Math.max(this.size[0], minWidth),
                        Math.max(this.size[1], minHeight)
                    ];
                }
            };

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

            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function(e, localPos, canvas) {
                const result = onMouseMove ? onMouseMove.apply(this, arguments) : undefined;
                
                this.updateCompareMode();
                
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
                
                this.setIsPointerDown(!!app.canvas.pointer_is_down);
                
                return result;
            };

            const onMouseLeave = nodeType.prototype.onMouseLeave;
            nodeType.prototype.onMouseLeave = function(e) {
                const result = onMouseLeave ? onMouseLeave.apply(this, arguments) : undefined;
                this.isPointerOver = false;
                this.sliderPos = null;
                
                this.setIsPointerDown(false);
                
                app.graph.setDirtyCanvas(true, false);
                return result;
            };

            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function(e, localPos, canvas) {
                const result = onMouseDown ? onMouseDown.apply(this, arguments) : undefined;
                
                this.updateCompareMode();
                
                if (this.compareMode === "click") {
                    this.setIsPointerDown(true);
                }
                
                return result;
            };

            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function(e, localPos, canvas) {
                const result = onMouseUp ? onMouseUp.apply(this, arguments) : undefined;
                
                this.setIsPointerDown(false);
                
                return result;
            };

            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                if (onDrawBackground) {
                    onDrawBackground.apply(this, arguments);
                }
            };

            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }
                
                if (!this.compareImages || this.compareImages.length < 2) return;

                const [nodeWidth, nodeHeight] = this.size;
                const imgA = this.compareImages[0];
                const imgB = this.compareImages[1];
                
                if (!imgA || !imgA.naturalWidth || !imgA.naturalHeight) return;
                if (!imgB || !imgB.naturalWidth || !imgB.naturalHeight) return;

                this.updateCompareMode();

                // Increased padding to clear buttons: dropdown (~30px) + buttons (72px) + margin (20px) = 122px
                const topPadding = 160;
                const sidePadding = 10;
                const bottomPadding = 10;

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

                // Clip everything below topPadding to prevent drawing behind widgets
                ctx.save();
                ctx.beginPath();
                ctx.rect(0, topPadding, nodeWidth, nodeHeight - topPadding);
                ctx.clip();

                // Draw image A (base layer)
                ctx.drawImage(imgA, dimsA.offsetX, dimsA.offsetY, dimsA.width, dimsA.height);

                let showImageB = false;
                let clipX = nodeWidth / 2;

                if (this.compareMode === "click") {
                    showImageB = this.isPointerDown;
                } else {
                    showImageB = this.isPointerOver && this.sliderPos !== null;
                    if (showImageB) {
                        clipX = this.sliderPos;
                    }
                }

                if (showImageB && this.compareMode === "slide") {
                    // Additional clipping for slider reveal
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
                    // Draw full image B
                    ctx.drawImage(imgB, dimsB.offsetX, dimsB.offsetY, dimsB.width, dimsB.height);
                }
                
                ctx.restore(); // Restore main clip region
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
                    img.onload = () => {
                        app.graph.setDirtyCanvas(true, false);
                    };
                    img.src = `data:image/png;base64,${base64Data}`;
                    node.compareImages.push(img);
                });
                node.imgs = [];
            }
        }

        app.graph.setDirtyCanvas(true, true);
        needsRefresh = false;
    }
}, 100);