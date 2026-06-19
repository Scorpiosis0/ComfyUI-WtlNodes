import math

# ─────────────────────────────────────────────────────────────────────────────
# Shared type sentinel — resolves to INT or FLOAT once x connects
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC = "NUMERIC"   # internal placeholder; JS swaps it live


# ─────────────────────────────────────────────────────────────────────────────
# Binary operations  (Add, Subtract, Multiply, Divide)
#   x           — socket (INT or FLOAT wire)
#   int_value   — INT slider widget, shown when INT connects to x
#   float_value — FLOAT slider widget, shown when FLOAT connects to x (default)
# JS hides whichever widget doesn't match the connected type.
# Python picks the active value via isinstance(x, int).
# ─────────────────────────────────────────────────────────────────────────────

class WtlAddC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x":           ("INT,FLOAT", {}),
                "int_value":   ("INT",   {"default": 0,   "min": -2**31, "max": 2**31-1, "step": 1}),
                "float_value": ("FLOAT", {"default": 0.0, "min": -1e9,   "max": 1e9,     "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, x, int_value, float_value):
        v = int_value if isinstance(x, int) else float_value
        return (x + v,)


class WtlSubtractC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x":           ("INT,FLOAT", {}),
                "int_value":   ("INT",   {"default": 0,   "min": -2**31, "max": 2**31-1, "step": 1}),
                "float_value": ("FLOAT", {"default": 0.0, "min": -1e9,   "max": 1e9,     "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, x, int_value, float_value):
        v = int_value if isinstance(x, int) else float_value
        return (x - v,)


class WtlMultiplyC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x":           ("INT,FLOAT", {}),
                "int_value":   ("INT",   {"default": 1,   "min": -2**31, "max": 2**31-1, "step": 1}),
                "float_value": ("FLOAT", {"default": 1.0, "min": -1e9,   "max": 1e9,     "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, x, int_value, float_value):
        v = int_value if isinstance(x, int) else float_value
        return (x * v,)


class WtlDivideC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x":           ("INT,FLOAT", {}),
                "int_value":   ("INT",   {"default": 1,   "min": -2**31, "max": 2**31-1, "step": 1}),
                "float_value": ("FLOAT", {"default": 1.0, "min": -1e9,   "max": 1e9,     "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, x, int_value, float_value):
        v = int_value if isinstance(x, int) else float_value
        if v == 0:
            raise ValueError("[WtlDivide] Division by zero")
        result = x / v
        if isinstance(x, int) and isinstance(v, int) and result == int(result):
            return (int(result),)
        return (float(result),)


# ─────────────────────────────────────────────────────────────────────────────
# Single-input operations  (Square, Square Root)
# ─────────────────────────────────────────────────────────────────────────────

class WtlSquareC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT,FLOAT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, a):
        result = a * a
        return (type(a)(result),)


class WtlSqrtC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT,FLOAT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, a):
        if a < 0:
            raise ValueError("[WtlSqrt] Square root of negative number")
        return (math.sqrt(a),)


# ─────────────────────────────────────────────────────────────────────────────
# Text Append  (dynamic: adds more slots as you connect)
# text_1 / text_2 use forceInput so they are always pure socket inputs —
# this keeps them in the same visual zone as the dynamically-added text_3+
# and prevents ordering issues where added slots appear above them.
# ─────────────────────────────────────────────────────────────────────────────

class WtlTextAppendC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"forceInput": True}),
                "text_2": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "separator": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, separator="", **kwargs):
        parts = []
        i = 1
        while f"text_{i}" in kwargs:
            v = kwargs[f"text_{i}"]
            if v is not None:
                parts.append(str(v))
            i += 1
        return (separator.join(parts),)


# ─────────────────────────────────────────────────────────────────────────────
# Type cast  (Int → Float  /  Float → Int)
# ─────────────────────────────────────────────────────────────────────────────

class WtlCastC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value":       ("INT,FLOAT", {}),
                "cast_to_int": ("BOOLEAN", {"default": False, "label_on": "→ INT", "label_off": "→ FLOAT"}),
            }
        }

    RETURN_TYPES = ("INT,FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION     = "execute"
    CATEGORY     = "WtlNodes/logic"

    def execute(self, value, cast_to_int):
        if cast_to_int:
            return (int(round(value)),)
        return (float(value),)


# ─────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "WtlAdd":        WtlAddC,
    "WtlSubtract":   WtlSubtractC,
    "WtlMultiply":   WtlMultiplyC,
    "WtlDivide":     WtlDivideC,
    "WtlSquare":     WtlSquareC,
    "WtlSqrt":       WtlSqrtC,
    "WtlTextAppend": WtlTextAppendC,
    "WtlCast":       WtlCastC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WtlAdd":        "Add",
    "WtlSubtract":   "Subtract",
    "WtlMultiply":   "Multiply",
    "WtlDivide":     "Divide",
    "WtlSquare":     "Square",
    "WtlSqrt":       "Square Root",
    "WtlTextAppend": "Text Append",
    "WtlCast":       "Int ↔ Float",
}
