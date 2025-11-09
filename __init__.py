# -*- coding: utf-8 -*-
# __init__.py  –  WtlNodes package entry point

from .image.saturation import NODE_CLASS_MAPPINGS as SAT_NODES
from .image.saturation import NODE_DISPLAY_NAME_MAPPINGS as SAT_DISPLAY

from .image.brightness import NODE_CLASS_MAPPINGS as BRIGHT_NODES
from .image.brightness import NODE_DISPLAY_NAME_MAPPINGS as BRIGHT_DISPLAY

from .image.contrast import NODE_CLASS_MAPPINGS as CONT_NODES
from .image.contrast import NODE_DISPLAY_NAME_MAPPINGS as CONT_DISPLAY

from .image.exposure import NODE_CLASS_MAPPINGS as EXP_NODES
from .image.exposure import NODE_DISPLAY_NAME_MAPPINGS as EXP_DISPLAY

from .image.temperature import NODE_CLASS_MAPPINGS as TEMP_NODES
from .image.temperature import NODE_DISPLAY_NAME_MAPPINGS as TEMP_DISPLAY

from .image.depth_dof import NODE_CLASS_MAPPINGS as DDOF_NODES
from .image.depth_dof import NODE_DISPLAY_NAME_MAPPINGS as DDOF_DISPLAY

from .image.latentp import NODE_CLASS_MAPPINGS as LATA_NODES
from .image.latentp import NODE_DISPLAY_NAME_MAPPINGS as LATA_DISPLAY

from .image.image_trans import NODE_CLASS_MAPPINGS as IMGT_NODES
from .image.image_trans import NODE_DISPLAY_NAME_MAPPINGS as IMGT_DISPLAY

from .mask.mask_trans import NODE_CLASS_MAPPINGS as MASKT_NODES
from .mask.mask_trans import NODE_DISPLAY_NAME_MAPPINGS as MASKT_DISPLAY

from .mask.mask_processor import NODE_CLASS_MAPPINGS as MPROC_NODES
from .mask.mask_processor import NODE_DISPLAY_NAME_MAPPINGS as MPROC_DISPLAY

from .cosine_scheduler.custom_scheduler import NODE_CLASS_MAPPINGS as CSCH_NODES
from .cosine_scheduler.custom_scheduler import NODE_DISPLAY_NAME_MAPPINGS as CSCH_DISPLAY

# -----------------------------------------------------------------
#  Combine all categories into the global mappings
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(SAT_NODES)
NODE_CLASS_MAPPINGS.update(BRIGHT_NODES)
NODE_CLASS_MAPPINGS.update(CONT_NODES)
NODE_CLASS_MAPPINGS.update(EXP_NODES)
NODE_CLASS_MAPPINGS.update(TEMP_NODES)
NODE_CLASS_MAPPINGS.update(DDOF_NODES)
NODE_CLASS_MAPPINGS.update(LATA_NODES)
NODE_CLASS_MAPPINGS.update(IMGT_NODES)
NODE_CLASS_MAPPINGS.update(MASKT_NODES)
NODE_CLASS_MAPPINGS.update(MPROC_NODES)
NODE_CLASS_MAPPINGS.update(CSCH_NODES)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(SAT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(BRIGHT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(CONT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(EXP_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(TEMP_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(DDOF_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(LATA_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(IMGT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(MASKT_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(MPROC_DISPLAY)
NODE_DISPLAY_NAME_MAPPINGS.update(CSCH_DISPLAY)

# -----------------------------------------------------------------
#  Aiohttp routes – now write to RAM instead of the filesystem
# -----------------------------------------------------------------
from aiohttp import web
import server

# -------------------------------------------------------------
#  Slider‑parameter route
# -------------------------------------------------------------
@server.PromptServer.instance.routes.post("/tgsz_dof_params")
async def dof_params(request):
    """Handle slider parameter updates – store them in RAM."""
    data = await request.json()
    node_id = str(data.get("node_id"))          # <-- ensure string key
    focus_depth = data.get("focus_depth")
    focus_range = data.get("focus_range")
    edge_fix = data.get("edge_fix")

    # Basic validation (optional)
    if not node_id:
        return web.json_response({"status": "error", "reason": "node_id missing"}, status=400)

    # Push the new values into the shared in‑memory dict
    from .image.depth_dof import _set_params
    _set_params(node_id, focus_depth, focus_range, edge_fix)

    print(f"[DOF] Params updated for node {node_id}: "
          f"focus={focus_depth}, range={focus_range}, edge={edge_fix}")
    return web.json_response({"status": "ok"})


# -------------------------------------------------------------
#  Button‑press route (Apply / Skip)
# -------------------------------------------------------------
@server.PromptServer.instance.routes.post("/tgsz_dof_control")
async def dof_control(request):
    """Handle **Apply** / **Skip** button clicks – set an in‑memory flag."""
    data = await request.json()
    node_id = str(data.get("node_id"))               # <-- ensure string key
    action = (data.get("action") or "").lower()

    if action not in ("apply", "skip"):
        return web.json_response(
            {"status": "error", "reason": "invalid action"}, status=400
        )

    # Store the flag; the node will notice it on the next poll.
    from .image.depth_dof import _set_flag
    _set_flag(node_id, action)

    print(f"[DOF] Flag '{action}' set for node {node_id}")
    return web.json_response({"status": "ok"})


# -------------------------------------------------------------
#  Slider‑parameter route
# -------------------------------------------------------------
@server.PromptServer.instance.routes.post("/tgsz_sat_params")
async def sat_params(request):
    """Handle slider parameter updates – store them in RAM."""
    data = await request.json()
    node_id = str(data.get("node_id"))          # <-- ensure string key
    saturation = data.get("saturation")

    # Basic validation (optional)
    if not node_id:
        return web.json_response({"status": "error", "reason": "node_id missing"}, status=400)

    # Push the new values into the shared in‑memory dict
    from .image.saturation import _set_params
    _set_params(node_id, saturation)

    print(f"[SAT] Params updated for node {node_id}: "
          f"focus={saturation}")
    return web.json_response({"status": "ok"})


# -------------------------------------------------------------
#  Button‑press route (Apply / Skip)
# -------------------------------------------------------------
@server.PromptServer.instance.routes.post("/tgsz_sat_control")
async def sat_control(request):
    """Handle **Apply** / **Skip** button clicks – set an in‑memory flag."""
    data = await request.json()
    node_id = str(data.get("node_id"))               # <-- ensure string key
    action = (data.get("action") or "").lower()

    if action not in ("apply", "skip"):
        return web.json_response(
            {"status": "error", "reason": "invalid action"}, status=400
        )

    # Store the flag; the node will notice it on the next poll.
    from .image.saturation import _set_flag
    _set_flag(node_id, action)

    print(f"[SAT] Flag '{action}' set for node {node_id}")
    return web.json_response({"status": "ok"})


# -----------------------------------------------------------------
#  Static web assets (unchanged)
# -----------------------------------------------------------------
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]