import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "RAMPreview.ImageDisplay",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RAMPreviewImage" ||
            nodeData.name === "saturationNode" ||
            nodeData.name === "DepthDOFNode" ||
            nodeData.name === "Exposure" ||
            nodeData.name === "Contrast" ||
            nodeData.name === "Brightness" ||
            nodeData.name === "ColorTemperatureNode"
        ) {
            
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function(message) {

                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                if (message?.ram_preview) {
                    // Store images
                    this.imgs = [];
                    
                    // Create image elements (ComfyUI will handle display)
                    message.ram_preview.forEach((base64Data, index) => {
                        const img = new Image();
                        img.onload = () => {};
                        img.src = `data:image/png;base64,${base64Data}`;
                        this.imgs.push(img);
                    });                    
                    // Force immediate redraw
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    }
});