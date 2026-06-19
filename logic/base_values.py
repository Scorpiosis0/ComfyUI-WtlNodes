# ─────────────────────────────────────────────────────────────────────────────
# Base value nodes  —  simple passthrough widgets that expose typed values
# on the graph so they can be wired anywhere.
# ─────────────────────────────────────────────────────────────────────────────

class WtlIntC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -2 ** 31,
                    "max":  2 ** 31 - 1,
                    "step": 1,
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, value):
        return (int(value),)


class WtlFloatC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1e9,
                    "max":  1e9,
                    "step": 0.01,
                    "round": False,
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, value):
        return (float(value),)


class WtlTextC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES  = ("text",)
    OUTPUT_NODE   = True
    FUNCTION      = "execute"
    CATEGORY      = "WtlNodes/logic"

    def execute(self, value):
        text = str(value)
        return {"ui": {"text_display": [text]}, "result": (text,)}


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "WtlInt":   WtlIntC,
    "WtlFloat": WtlFloatC,
    "WtlText":  WtlTextC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WtlInt":   "Int",
    "WtlFloat": "Float",
    "WtlText":  "Text",
}
