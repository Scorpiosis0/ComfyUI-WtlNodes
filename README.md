# ComfyUI-WtlNodes

A professional-grade custom node pack for ComfyUI built around accuracy, control, and a fast iterative workflow.

Color grading nodes use proper color science — HSV-space adjustments, physically-based exposure in EV stops, Kelvin-accurate color temperature with luminance preservation, and tone-range isolation with Gaussian feathering. These are not simple RGB multipliers; the math is there to produce results that hold up under close inspection.

Live preview is built into the majority of nodes: sliders update in real time while the workflow is paused, letting you dial in values before committing. Apply or skip without re-running the full graph. Batch images are handled one by one or all at once depending on your choice.

On the effects side, nodes are designed for granular control rather than one-knob shortcuts. Retro film simulation, lens optics, and stylistic filters each expose the individual parameters that actually matter — intensity curves, physical shape options, per-channel behavior, threshold ranges — so you can craft an exact look rather than accept a preset.

The same philosophy carries through to transforms, masks, latent utilities, and scheduling: sensible defaults, full range of options, no black boxes.

---

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Scorpiosis0/ComfyUI-WtlNodes.git
```

Or download the ZIP and extract into `ComfyUI/custom_nodes/`.

---

## Node List

### Color Adjustment

These nodes adjust image color properties and support live interactive preview via `apply_type`.

| Node | Description |
|---|---|
| **Brightness** | Multiplies pixel values to make the image brighter or darker. Range: -100 to 100. |
| **Contrast** | Adjusts contrast around a 0.5 pivot point. Range: -100 to 100. |
| **Exposure** | Adjusts exposure in stops using `2^exposure` multiplication. Range: -10 to 10 EV. |
| **Saturation (HSV)** | Adjusts color saturation in HSV color space. Range: -100 to 100. |
| **Hue** | Rotates the hue of the image in HSV space. Range: 0 to 360 degrees. |
| **Temperature (Tanner Helland's algorithm)** | Shifts color temperature in Kelvin (1000K = warm orange, 6500K = neutral, 40000K = cold blue). |
| **Highlight & Shadow** | Independently brighten/darken highlights and shadows. Includes midpoint and feather radius controls for smooth transitions. |

**`apply_type` options (shared by all color nodes):**
- `none` — apply the node value from the widget directly, no interactive preview
- `auto_apply` — same as `none`, alias
- `apply_all` — pause workflow, show live preview, wait for Apply/Skip button

---

### Depth of Field

| Node | Description |
|---|---|
| **Depth of Field (DOF)** | Applies Gaussian blur using a depth map. Set `focus_depth` (0 = background, 1 = foreground), `focus_range` (falloff size), `hard_focus_range` (sharp zone around focus), `blur_strength`, and `edge_fix` to smooth out depth edge artifacts. Outputs image + blur mask. |
| **Camera Depth of Field (WIP)** | Advanced DOF with bokeh shape (circle / hexagon / octagon), highlight bloom, depth-aware multi-level blur, and an in-focus mask fix for edge cleanup. Outputs image + blur mask + in-focus mask + out-of-focus mask + border mask. |

---

### Image Effects

All effects below support live preview via `apply_type`.

| Node | Description |
|---|---|
| **Dither** | Reduces color depth using dithering. Methods: `none`, `bayer`, `arithmetic_add`, `blue_noise`. Controls per-channel color levels (R/G/B) and dither scale. |
| **Film Grain** | Adds realistic film grain. Controls: `intensity`, `grain_size`, and `monochrome` toggle. |
| **Chromatic Aberration** | Shifts red and blue channels independently by pixel offset and scale. Supports center point and falloff for lens-like radial distortion. |
| **Film Artifacts** | Adds retro film damage: scratches, dust, hair, light leaks, and vignette. All elements are controlled by density and size sliders plus a seed for reproducibility. Requires a pre-generated cache file (`film_artifacts_cache.pkl`). |
| **Image Filters** | Applies stylistic filters: `b&w`, `sepia`, `duotone`, `invert`, `cartoon`, `sketch`, `neon`, `high_contrast`, `emboss`, `infrared`. Includes `strength`, `edge_threshold`, and neon-specific controls. |
| **CRT TV Effect** | Simulates a CRT monitor with scanlines, barrel curvature, chromatic aberration, halation (glow), phosphor dots, noise, and vignette. |
| **ASCII Effect** | Converts the image to ASCII art. Controls: character set, font name, char size, character spacing, RGB channel weights for brightness mapping, background color, bold/italic toggles. |

---

### Image Transform

| Node | Description |
|---|---|
| **Image Resize** | Resize by absolute dimensions or multiplier. Fit modes: `crop`, `adjust`, `fit`. Supports multiple interpolation methods and bg fill color. |
| **Image Rotation** | Rotate by degrees (-360 to 360). Fit modes: `crop`, `fit`, `adjust`, `none`. |
| **Image Zoom** | Zoom in/out with optional X/Y translation. Useful for crop-zoom effects. |
| **Image Translation** | Shift image horizontally/vertically by pixel offset. |

---

### Mask

| Node | Description |
|---|---|
| **Mask Processor** | Dilate (positive) or erode (negative) a mask, then feather the edges. |
| **Mask Filter** | Filter a batch of masks by area size. Keeps masks `above_x`, `below_x`, or `between_x_y` a pixel-area threshold. Useful for removing small/large detected regions. |
| **Mask Resize** | Resize a mask with the same options as Image Resize. Includes an `enhanced_visibility` toggle for easier preview. |
| **Mask Rotation** | Rotate a mask. Same options as Image Rotation. |
| **Mask Zoom** | Zoom and translate a mask. |
| **Mask Translation** | Shift a mask horizontally/vertically. |

---

### Latent

| Node | Description |
|---|---|
| **Empty Latent (Advanced)** | Create an empty latent with aspect ratio presets (1:1, 3:2, 4:3, 5:3, 16:9, 16:10, 21:9, 32:9) and portrait/landscape toggle. Falls back to manual width/height. All resolutions are multiples of 64. Returns latent + latent dimensions + pixel dimensions. |
| **Latent Noise Injector** | Injects procedural noise into a latent, scaled by the first sigma value. Noise types: `Gaussian`, `White`, `Perlin`, `Simplex`, `Worley`, `Voronoi`. Controls: `noise_multiplier` and `scale` (frequency). |
| **Tiled Sampler (Custom Advanced)** | Samples large latents in tiles to reduce VRAM usage. Seam blending via cosine masks. Controls: tile factor, context size (pixels of surrounding image the model sees), seam flat width, and seam feather. Slot in the `sampling/custom_sampling` category. |

---

### Sigma / Scheduler

| Node | Description |
|---|---|
| **Dual Ease Cosine Scheduler** | Custom sigma schedule with independent easing at the top (`rho_start`) and bottom (`rho_end`) of the curve. Tested with FID and ablation to find good default values. Works with any sampler that accepts SIGMAS. |
| **Sigma Visualizer** | Renders a sigma schedule as a chart image, displayed in the node preview. Output only, no image is saved to disk. |

---

### Preview / Compare

| Node | Description |
|---|---|
| **RAM Preview Image** | Previews images directly in the node without saving to disk. Useful for inspecting intermediate results. |
| **RAM Image Compare** | Side-by-side image comparison with `slide` (drag divider) or `click` (click to toggle) modes. No disk I/O. |
| **Blind Comparer** | Tournament bracket for comparing multiple images blindly. Connect any number of images; vote Left or Right each round. The bracket continues until a winner is found. |

---

### Logic

Simple utility nodes for connecting numeric and text values.

| Node | Description |
|---|---|
| **Int** | Outputs an integer constant. |
| **Float** | Outputs a float constant. |
| **Text** | Outputs a text string with multiline support. |
| **Add** | Adds an int or float value to the input. |
| **Subtract** | Subtracts an int or float value from the input. |
| **Multiply** | Multiplies the input by an int or float value. |
| **Divide** | Divides the input by an int or float value. Raises an error on division by zero. |
| **Square** | Returns `x²` of the input. |
| **Square Root** | Returns `√x` of the input. |
| **Text Append** | Concatenates two or more strings with an optional separator. Slots expand dynamically as you connect inputs. |
| **Int ↔ Float** | Casts between INT and FLOAT. |
