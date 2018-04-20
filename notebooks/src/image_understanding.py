from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from notebooks.src.understanding_images.d3 import pixels_3d, pixels_image_3d


# TODO: Check out matplotlib voxels in 3D plotting. They can plot cubes and probably more efficiently.


def plot_art(n=1, no_axis=True):
    rgb_image = np.load(str(Path("notebooks", "src", "understanding_images", "data", "art{}.npz".format(n))))
    pixels_image_3d(
        rgb_image=rgb_image,
        no_axis=no_axis
    )


def plot_color_scales(
        scales="red,lime,blue,yellow,magenta,cyan,white",
        n_shapes=10, x_spread=3, y_spread=0, z_spread=3.5):
    if isinstance(scales, str):
        scales = scales.split(",")

    # Bases of each scale
    color_bases = np.array([to_rgb(val) for val in scales])

    # Make pixels and set positions
    positions = []
    pixel_colors = []
    for color_nr, color_base in enumerate(color_bases):
        for val_nr, val in enumerate(np.linspace(1, 0, n_shapes)):
            positions.append((color_nr * x_spread, color_nr * y_spread + val_nr, color_nr * z_spread))
            pixel_colors.append(color_base * val)

    # Plot 3D pixels
    pixels_3d(
        positions=positions,
        pixel_colors=pixel_colors,
        camera_position="x",
        no_axis=True,
        linewidths=0.1
    )


def plot_single_pixel(color):
    color = np.array(to_rgb(color))
    pixels_3d(
        positions=[(0, 0, 0)],
        pixel_colors=[color],
        camera_position="x",
        no_axis=True,
        linewidths=0.1
    )


if __name__ == "__main__":
    plt.close("all")

    plot_color_scales()
