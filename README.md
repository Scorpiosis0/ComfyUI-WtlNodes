# ComfyUI-WtlNodes

A professional-grade custom node pack for ComfyUI built around accuracy, control, and a fast iterative workflow.

Color grading nodes use proper color science — HSV-space adjustments, physically-based exposure in EV stops, Kelvin-accurate color temperature with luminance preservation, and tone-range isolation with Gaussian feathering. These are not simple RGB multipliers; the math is there to produce results that hold up under close inspection.

Live preview is built into the majority of nodes: sliders update in real time while the workflow is paused, letting you dial in values before committing. Apply or skip without re-running the full graph.

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

## Live Preview System

Most nodes support an `apply_type` dropdown that controls how the node interacts with the workflow:

| Mode | Behaviour |
|---|---|
| `none` | Pauses the workflow per image in the batch, shows a live RAM preview, and waits for **Apply** or **Skip** per image. Sliders update the preview in real time. |
| `auto_apply` | Runs directly with the current widget values — no interactive pause, no preview loop. |
| `apply_all` | Pauses once for the whole batch, shows a live preview, and waits for a single **Apply** or **Skip** decision that applies to all images. |

---

## Node List

### Color Adjustment

| Node | Description |
|---|---|
| **Brightness** | Multiplies pixel values. Range: −100 to 100. |
| **Contrast** | Adjusts contrast around a 0.5 pivot point. Range: −100 to 100. |
| **Exposure** | Adjusts exposure in stops (`2^EV` multiplication). Range: −10 to 10 EV. |
| **Saturation (HSV)** | Adjusts color saturation in HSV space. Range: −100 to 100. |
| **Hue** | Rotates hue in HSV space. Range: 0–360°. |
| **Temperature** | Shifts color temperature in Kelvin using Tanner Helland's algorithm with luminance preservation. 1000 K = warm orange, 6500 K = neutral, 40000 K = cold blue. |
| **Highlight & Shadow** | Independently brightens/darkens highlights and shadows with midpoint and feather radius controls. |

All color nodes support the Live Preview System via `apply_type`.

---

### Depth of Field

| Node | Description |
|---|---|
| **Camera Depth of Field** | Full camera-simulation DOF with 8-level depth-graduated blur, physically shaped bokeh kernels (circle / hexagon / octagon), highlight bloom, and an `in_focus_mask_fix` dilation for edge cleanup. `preview_mode` switches the live preview between the blur mask, in-focus mask, or blurred image. Outputs image + four masks. |

---

### Image Effects

All effects support live preview via `apply_type`.

| Node | Description |
|---|---|
| **CRT TV Effect** | Simulates a CRT monitor with a full 12-step effect chain: phosphor tint, defocus, phosphor dots, halation, bloom, scanlines (`cos²` periodic darkening), beam sweep (stylistic bright trace with per-column luma modulation), noise, barrel curvature, chromatic aberration, and vignette. Includes a `grayscale` toggle for clean phosphor tint looks. |
| **Film Grain** | Adds realistic film grain. Controls: `intensity`, `grain_size`, and `monochrome` toggle. |
| **Chromatic Aberration** | Shifts red and blue channels by pixel offset and scale. Supports center point and falloff for radial lens-like distortion. |
| **Film Artifacts** | Adds retro film damage: scratches, dust, hair, light leaks, and vignette. All controlled by density and size sliders plus a seed. |
| **Image Filters** | Stylistic filters: `b&w`, `sepia`, `duotone`, `invert`, `cartoon`, `sketch`, `neon`, `high_contrast`, `emboss`, `infrared`. Includes `strength`, `edge_threshold`, and neon-specific controls. |
| **Dither** | Reduces color depth using dithering. Methods: `none`, `bayer`, `arithmetic_add`, `blue_noise`. Per-channel level controls and dither scale. |
| **ASCII Effect** | Converts the image to ASCII art. Controls: character set, font, char size, spacing, RGB weights, background color, bold/italic toggles. |

---

### Image Transform

| Node | Description |
|---|---|
| **Image Resize** | Resize by absolute dimensions or multiplier. Fit modes: `crop`, `adjust`, `fit`. Multiple interpolation methods and bg fill color. |
| **Image Rotation** | Rotate by degrees (−360 to 360). Fit modes: `crop`, `fit`, `adjust`, `none`. |
| **Image Zoom** | Zoom in/out with optional X/Y translation. |
| **Image Translation** | Shift image horizontally/vertically by pixel offset. |

---

### Mask

| Node | Description |
|---|---|
| **Mask Processor** | Dilate (positive) or erode (negative) a mask, then feather the edges with Gaussian blur. |
| **Mask Filter** | Filter a batch of masks by pixel area. Keep masks `above_x`, `below_x`, or `between_x_y` a threshold. |
| **Mask Resize** | Resize a mask with the same options as Image Resize. Includes `enhanced_visibility` (red background preview). |
| **Mask Rotation** | Rotate a mask. Same options as Image Rotation. |
| **Mask Zoom** | Zoom and translate a mask. |
| **Mask Translation** | Shift a mask horizontally/vertically. |
| **RAM Preview Mask** | Displays a mask in the node preview panel without saving to disk. Greyscale display, RAM-only. |

---

### Latent

| Node | Description |
|---|---|
| **Empty Latent (Advanced)** | Creates an empty latent with aspect ratio presets (1:1, 3:2, 4:3, 5:3, 16:9, 16:10, 21:9, 32:9) and portrait/landscape toggle, or manual width/height. `latent_compression` slider (default 8) controls the spatial downscale factor — adjust for VAEs that differ from the standard 8× compression. Returns latent + latent dimensions + pixel dimensions. |
| **Latent Noise Injector** | Injects procedural noise into a latent, scaled by the first sigma value. Noise types: Gaussian, White, Perlin, Simplex, Worley, Voronoi. Controls: `noise_multiplier` and `scale` (spatial frequency). |
| **Tiled Sampler (Custom Advanced)** | Samples large latents in overlapping tiles to reduce VRAM usage. Seam blending via cosine-feathered masks with a dedicated seam-fix pass. Controls: tile factor, context size, seam flat width, seam feather, and seam fix sigma. Lives in `sampling/custom_sampling`. |

---

### Sigma / Scheduler

| Node | Description |
|---|---|
| **Dual Ease Cosine Scheduler** | Custom sigma schedule with independent easing at the start (`rho_start`) and end (`rho_end`) of the curve. Works with any sampler that accepts SIGMAS. |
| **Sigma Visualizer** | Renders a sigma schedule as a chart displayed in the node preview. No file is saved to disk. |

---

### Preview / Compare

| Node | Description |
|---|---|
| **RAM Preview Image** | Previews images in the node without saving to disk. Useful for inspecting intermediate results in a workflow. |
| **RAM Image Compare** | Side-by-side comparison with `slide` (drag divider) or `click` (toggle) modes. No disk I/O. |
| **Blind Comparer** | Tournament bracket for comparing multiple images blindly. Connect any number of images; vote Left or Right each round until a winner is found. |

---

### Logic

Simple utility nodes for connecting numeric and text values.

| Node | Description |
|---|---|
| **Int** | Outputs an integer constant. |
| **Float** | Outputs a float constant. |
| **Text** | Outputs a text string with multiline support. |
| **Add** | Adds a value to the input. |
| **Subtract** | Subtracts a value from the input. |
| **Multiply** | Multiplies the input by a value. |
| **Divide** | Divides the input by a value. Raises an error on division by zero. |
| **Square** | Returns `x²` of the input. |
| **Square Root** | Returns `√x` of the input. |
| **Text Append** | Concatenates strings with an optional separator. Slots expand dynamically. |
| **Int ↔ Float** | Casts between INT and FLOAT. |
