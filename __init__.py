# __init__.py  –  WtlNodes package entry point
import importlib
import server
from aiohttp import web

# Define submodules to import
SUBMODULES = [
    "image.saturation",
    "image.brightness",
    "image.contrast",
    "image.exposure",
    "image.temperature",
    "image.depth_dof",
    "image.latent_a",
    "image.image_trans",
    "mask.mask_trans",
    "mask.mask_processor",
    "sigma.dual_ease_cosine_scheduler",
    "sigma.sigma_visualizer",
    "image.ram_preview_image",
]

# Initialize global mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Dynamically import and update the mappings
for submodule in SUBMODULES:
    module = importlib.import_module(f".{submodule}", package=__package__)
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

# Create registry for node handlers
NODE_HANDLERS = {
    "dof": {
        "module": ".image.depth_dof",
        "params": ["focus_depth", "focus_range", "edge_fix", "hard_focus_range"]
    },
    "sat": {
        "module": ".image.saturation",
        "params": ["saturation"]
    },
    "exp": {
        "module": ".image.exposure",
        "params": ["exposure"]
    },
    "con": {
        "module": ".image.contrast",
        "params": ["contrast"]
    },
    "bri": {
        "module": ".image.brightness",
        "params": ["brightness"]
    },
    "tem": {
        "module": ".image.temperature",
        "params": ["temperature"]
    }
}

# Slider‑parameter route
@server.PromptServer.instance.routes.post("/tgsz_params")
async def tgsz_params(request):
    """Handle slider parameter updates – store them in RAM."""
    data = await request.json()
    node_id = str(data.get("node_id"))
    node_type = data.get("node_type")
    
    if not node_id or not node_type:
        return web.json_response({"status": "error", "reason": "node_id or node_type missing"}, status=400)
    
    if node_type not in NODE_HANDLERS:
        return web.json_response({"status": "error", "reason": "invalid node_type"}, status=400)
    
    handler = NODE_HANDLERS[node_type]
    
    # Dynamically import the module
    from importlib import import_module
    module = import_module(handler["module"], package=__package__)
    
    # Extract only params this node type needs
    params = [data.get(k) for k in handler["params"]]
    module._set_params(node_id, *params)
    
    print(f"[{node_type.upper()}] Params updated for node {node_id}")
    return web.json_response({"status": "ok"})

# Button‑press route (Apply / Skip)
@server.PromptServer.instance.routes.post("/tgsz_control")
async def tgsz_control(request):
    """Handle **Apply** / **Skip** button clicks – set an in‑memory flag."""
    data = await request.json()
    node_id = str(data.get("node_id"))
    node_type = data.get("node_type")
    action = (data.get("action") or "").lower()
    
    if not node_id or not node_type:
        return web.json_response({"status": "error", "reason": "node_id or node_type missing"}, status=400)
    
    if action not in ("apply", "skip"):
        return web.json_response({"status": "error", "reason": "invalid action"}, status=400)
    
    if node_type not in NODE_HANDLERS:
        return web.json_response({"status": "error", "reason": "invalid node_type"}, status=400)
    
    handler = NODE_HANDLERS[node_type]
    
    # Dynamically import module
    from importlib import import_module
    module = import_module(handler["module"], package=__package__)
    
    # Call _set_flag on appropriate module
    module._set_flag(node_id, action)
    
    print(f"[{node_type.upper()}] Flag '{action}' set for node {node_id}")
    return web.json_response({"status": "ok"})

# Static web assets (unchanged)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]