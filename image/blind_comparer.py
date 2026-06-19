import torch
import threading
import time
import random
import base64
import io
import numpy as np
import server
from aiohttp import web
from PIL import Image as PILImage

# ─────────────────────────────────────────────────────────────────────────────
# In-memory control store
# ─────────────────────────────────────────────────────────────────────────────
_CONTROL_STORE: dict[str, dict] = {}
_CONTROL_LOCK = threading.Lock()


def _set_flag(node_id: str, flag: str) -> None:
    with _CONTROL_LOCK:
        _CONTROL_STORE.setdefault(node_id, {}).setdefault("flags", {})[flag] = True


def _check_and_clear_flag(node_id: str, flag: str) -> bool:
    with _CONTROL_LOCK:
        entry = _CONTROL_STORE.get(node_id)
        if not entry:
            return False
        flags = entry.get("flags", {})
        if flags.get(flag):
            flags[flag] = False
            return True
        return False


def _clear_all(node_id: str) -> None:
    with _CONTROL_LOCK:
        _CONTROL_STORE.pop(node_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP routes
# ─────────────────────────────────────────────────────────────────────────────

@server.PromptServer.instance.routes.post("/wtl_bracket_vote")
async def wtl_bracket_vote(request):
    data    = await request.json()
    node_id = str(data.get("node_id", ""))
    action  = str(data.get("action", "")).lower()
    if not node_id or action not in ("left", "right"):
        return web.json_response({"status": "error"}, status=400)
    _set_flag(node_id, action)
    return web.json_response({"status": "ok"})


@server.PromptServer.instance.routes.post("/wtl_bracket_skip")
async def wtl_bracket_skip(request):
    data    = await request.json()
    node_id = str(data.get("node_id", ""))
    if node_id:
        _set_flag(node_id, "skip")
    return web.json_response({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_b64(t: torch.Tensor) -> str:
    arr = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG", compress_level=0)
    return base64.b64encode(buf.getvalue()).decode()


def _send_match(img_a, img_b, orig_a, orig_b, round_num, uid):
    server.PromptServer.instance.send_sync(
        "executed",
        {
            "node": uid,
            "output": {
                "ram_preview": [
                    _tensor_to_b64(img_a[0:1]),
                    _tensor_to_b64(img_b[0:1]),
                ],
                "bracket_match": {
                    "left_label":  f"image_{orig_a + 1}",
                    "right_label": f"image_{orig_b + 1}",
                    "round":       round_num,
                },
            },
            "prompt_id": None,
        }
    )


def _send_champion(img_champ, orig, uid):
    server.PromptServer.instance.send_sync(
        "executed",
        {
            "node": uid,
            "output": {
                "ram_preview":      [_tensor_to_b64(img_champ[0:1])],
                "bracket_champion": f"image_{orig + 1}",
            },
            "prompt_id": None,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bracket runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_bracket(images: list, uid: str, node) -> int | None:
    """
    Positional knockout bracket on a shuffled pool of original indices.

    Each round: pair pool[0]vs[1], pool[2]vs[3], ...
    Odd image at the end auto-advances (bye) — no vote shown.
    Returns the winning original index, or None if aborted.
    """
    pool = list(range(len(images)))
    random.shuffle(pool)

    round_num = 0
    while len(pool) > 1:
        next_pool = []
        i = 0
        while i < len(pool):
            if i + 1 < len(pool):
                orig_a = pool[i]
                orig_b = pool[i + 1]
                _send_match(images[orig_a], images[orig_b], orig_a, orig_b, round_num, uid)
                winner = node._wait_for_vote(uid, orig_a, orig_b)
                if winner is None:
                    return None
                next_pool.append(winner)
                i += 2
            else:
                # Lone image — auto-advance silently, no vote
                next_pool.append(pool[i])
                i += 1
        pool = next_pool
        round_num += 1

    return pool[0]


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class BlindComparerC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE  = True
    FUNCTION     = "run_bracket"
    CATEGORY     = "WtlNodes/image"
    DESCRIPTION  = (
        "Blind knockout image comparer. Connect images to auto-expand slots. "
        "Vote Left or Right each match. Odd images get a bye. Champion shown at end."
    )

    def run_bracket(self, unique_id=None, **kwargs):
        uid = str(unique_id) if unique_id else "0"
        _clear_all(uid)

        images: list[torch.Tensor] = []
        i = 1
        while f"image_{i}" in kwargs:
            images.append(kwargs[f"image_{i}"])
            i += 1

        n = len(images)
        if n == 0:
            return {}
        if n == 1:
            _send_champion(images[0], 0, uid)
            return {}

        champion = _run_bracket(images, uid, self)
        if champion is not None:
            _send_champion(images[champion], champion, uid)
        return {}

    @staticmethod
    def _wait_for_vote(uid: str, orig_a: int, orig_b: int):
        while True:
            if _check_and_clear_flag(uid, "left"):
                return orig_a
            if _check_and_clear_flag(uid, "right"):
                return orig_b
            if _check_and_clear_flag(uid, "skip"):
                return None
            time.sleep(0.05)


NODE_CLASS_MAPPINGS        = {"BlindComparer": BlindComparerC}
NODE_DISPLAY_NAME_MAPPINGS = {"BlindComparer": "Blind Comparer"}