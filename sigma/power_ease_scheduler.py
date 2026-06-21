import torch
import numpy as np

class PowerEaseSchedulerC:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "sigma_max": ("FLOAT", {
                    "default": 14.614642,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.02917,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01
                }),
                "power": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "WtlNodes/sigma"

    def get_sigmas(self, steps, sigma_max, sigma_min, power, model=None):
        if model is not None:
            ms = model.get_model_object("model_sampling")
            sigma_max = float(ms.sigma_max)
            sigma_min = float(ms.sigma_min)

        # t from 0 to 1, sigma = sigma_max - (sigma_max - sigma_min) * t^power
        # At t=0: sigma_max, at t=1: sigma_min
        t = np.linspace(0, 1, steps, dtype=np.float64)
        sigmas = sigma_max - (sigma_max - sigma_min) * (t ** power)

        # Enforce strictly decreasing with minimum spacing (SDE fix)
        min_spacing = 1e-4
        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i - 1] - min_spacing:
                sigmas[i] = sigmas[i - 1] - min_spacing

        sigmas = np.append(sigmas, 0.0)
        return (torch.FloatTensor(sigmas),)

NODE_CLASS_MAPPINGS = {"PowerEaseScheduler": PowerEaseSchedulerC}
NODE_DISPLAY_NAME_MAPPINGS = {"PowerEaseScheduler": "Power Ease Scheduler"}
