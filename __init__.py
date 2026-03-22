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
    "image.camera_depth_dof",
    "latent.latent_a",
    "latent.noise_injector",
    "image.highlight_shadow",
    "image.hue",
    "image.image_resize",
    "image.image_rotation",
    "image.image_zoom",
    "image.image_translation",
    "mask.mask_resize",
    "mask.mask_rotation",
    "mask.mask_zoom",
    "mask.mask_translation",
    "mask.mask_processor",
    "mask.mask_filter",
    "sigma.dual_ease_cosine_scheduler",
    "sigma.sigma_visualizer",
    "image.ram_preview_image",
    "image.dithering",
    "image.ram_compare",
    "image.ascii",
    "image.film_grain",
    "image.chromatic_aberration",
    "image.film_artifact",
    "image.image_filter",
    "image.crt",
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
        "params": ["focus_depth", "focus_range", "edge_fix", "hard_focus_range","blur_strength"]
    },
    "cdof": {
        "module": ".image.camera_depth_dof",
        "params": ["focal_point", "focus_falloff", "edge_fix", "focal_plane", "blur_strength", "in_focus_mask_fix", "bokeh_shape", "highlight_factor", "highlight_threshold_low", "highlight_threshold_high", "depth_aware_blur", "blur_fixed_edge"]
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
    },
    "hue": {
        "module": ".image.hue",
        "params": ["hue"]
    },
    "mfl": {
        "module": ".mask.mask_filter",
        "params": ["area_x", "area_y", "keep"]
    },
    "has": {
        "module": ".image.highlight_shadow",
        "params": ["shadow_adjustment", "highlight_adjustment", "midpoint", "feather_radius"]
    },
    "mpr": {
        "module": ".mask.mask_processor",
        "params": ["dilate_erode", "feather"]
    },
    "iro": {
        "module": ".image.image_rotation",
        "params": ["rotate", "interpolation", "fit_mode", "bg_color"]
    },
    "ire": {
        "module": ".image.image_resize",
        "params": ["resize_by", "width", "height", "multiplier", "interpolation", "fit_mode", "bg_color"]
    },
    "izo": {
        "module": ".image.image_zoom",
        "params": ["zoom", "interpolation", "translate_x", "translate_y", "bg_color"]
    },
    "itr": {
        "module": ".image.image_translation",
        "params": ["translate_x", "translate_y", "bg_color"]
    },
    "mro": {
        "module": ".mask.mask_rotation",
        "params": ["rotate", "interpolation", "fit_mode", "enhanced_visibility"]
    },
    "mre": {
        "module": ".mask.mask_resize",
        "params": ["resize_by", "width", "height", "multiplier", "interpolation", "fit_mode", "enhanced_visibility"]
    },
    "mzo": {
        "module": ".mask.mask_zoom",
        "params": ["zoom", "interpolation", "translate_x", "translate_y", "enhanced_visibility"]
    },
    "mtr": {
        "module": ".mask.mask_translation",
        "params": ["translate_x", "translate_y", "enhanced_visibility"]
    },
    "dit": {
        "module": ".image.dithering",
        "params": ["dither_method", "r_levels", "g_levels", "b_levels", "dither_scale"]
    },
    "asc": {
        "module": ".image.ascii",
        "params": ["red_weight", "green_weight", "blue_weight", "char_set", "char_size", "background", "font_name", "bold", "italic", "spacing"]
    },
    "fgr": {
        "module": ".image.film_grain",
        "params": ["intensity", "grain_size", "monochrome"]
    },
    "chromatic": {
        "module": ".image.chromatic_aberration",
        "params": ["offset_x", "offset_y", "red_scale", "blue_scale", "center_x", "center_y", "falloff"]
    },
    "fil": {
        "module": ".image.film_artifact",
        "params": ["intensity", "scratch_density", "scratch_max_length", "scratch_max_width", "dust_density", "dust_max_size", "hair_density", "hair_max_length", "light_leak_intensity", "vignette_strength", "seed"]
    },
    "iflt": {
        "module": ".image.image_filter",
        "params": ["filter_type", "strength", "edge_threshold", "neon_hue", "neon_blur"]
    },
    "crt": {
        "module": ".image.crt",
        "params": ["scanline_intensity", "scanline_width", "curvature", "chromatic_aberration", "halation", "phosphor_dots", "noise", "vignette"]
    },
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

# Button‑press route (Apply / Skip / Apply Again)
@server.PromptServer.instance.routes.post("/tgsz_control")
async def tgsz_control(request):
    """Handle **Apply** / **Skip** / **Apply Again** button clicks – set an in‑memory flag."""
    data = await request.json()
    node_id = str(data.get("node_id"))
    node_type = data.get("node_type")
    action = (data.get("action") or "").lower()
    
    if not node_id or not node_type:
        return web.json_response({"status": "error", "reason": "node_id or node_type missing"}, status=400)
    
    # UPDATED: Now accepts "apply", "skip", AND "apply_again"
    if action not in ("apply", "skip", "apply_again"):
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

# Get processing time route
@server.PromptServer.instance.routes.get("/tgsz_time/{node_type}/{node_id}")
async def tgsz_time(request):
    """Get the processing time for a node."""
    node_type = request.match_info.get("node_type")
    node_id = request.match_info.get("node_id")
    
    if not node_id or not node_type:
        return web.json_response({"status": "error", "reason": "node_id or node_type missing"}, status=400)
    
    if node_type not in NODE_HANDLERS:
        return web.json_response({"status": "error", "reason": "invalid node_type"}, status=400)
    
    handler = NODE_HANDLERS[node_type]
    
    # Dynamically import module
    from importlib import import_module
    module = import_module(handler["module"], package=__package__)
    
    # Get processing time if function exists
    if hasattr(module, '_get_processing_time'):
        processing_ms, complete = module._get_processing_time(node_id)
        return web.json_response({
            "status": "ok", 
            "processing_time_ms": processing_ms,
            "complete": complete
        })
    else:
        return web.json_response({"status": "ok", "processing_time_ms": 0, "complete": False})

# Static web assets (unchanged)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]