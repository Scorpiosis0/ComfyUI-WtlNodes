Hi, I made this node pack because I wanted :)

**Nodes list ⬇️**

Dual Ease Cosine Scheduler:

This node is a custom scheduler I made that needs further testing. It's been tested using FID (fréchet inception distance) along ablation to find the rought best default values.

Settings ⚙️:
- steps : controls how many steps will be made
- sigma_max : controls the maximum sigma (max noise)
- min_sigma : controls the minimum sigma (min noise)
- rho_start : controls the strenght of the easing curve at the top
- rho_end : controls the strenght of the easing curve at the bottom

Visualize Sigma Schedule :

This node will allow you to visualize sigma curves.

Empty Latent (Advanced) :

This node will give you access to premade ratios resoltions sets along with regular manual resoltions input and a portrait mode switch.

Settings ⚙️:
- use_ratio : This setting allows you to switch between Manual Mode and Ratio Mode. Ratio Mode has predefined ratios and resolutions and Manual Mode allows manual resolution input.
- portait_landscape : This allow you to inverse width and height hence making portraits or landscape while using Ratio Mode.
- width & height : When in Manual Mode those two settings allow to set the desired latent size you want.
- ratio : When in Ratio Mode this setting allows to set the desired ratio you want.
- resolution : When in Ratio Mode this setting allows to set the desired resolution you want based on the ratio, all resoltion are 64 multiples to avoid genration artifacts.
- batch : Controls how many latents are created at once.
