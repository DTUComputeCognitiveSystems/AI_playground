from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from matplotlib.colors import to_rgb
import importlib

from notebooks.src.understanding_images import d3

importlib.reload(d3)


def plot_art(n=1, no_axis=True):
    rgb_image = np.load(str(Path("notebooks", "src", "understanding_images", "data", "art{}.npz".format(n))))
    d3.pixels_image_3d(
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
    return d3.pixels_3d(
        positions=positions,
        pixel_colors=pixel_colors,
        camera_position="xy",
        no_axis=True,
        linewidths=0.1,
    )


def plot_single_pixel(color, ax=None):
    color = np.array(to_rgb(color))
    output = d3.pixels_3d(
        positions=[(0, 0, 0)],
        pixel_colors=[color],
        camera_position="x",
        no_axis=True,
        linewidths=0.1,
        ax=ax,
    )
    return output


def rgb_sliders():
    sliders = []
    for text in ["Red", "Green", "Blue"]:
        sliders.append(widgets.FloatSlider(
            value=1.0,
            min=0,
            max=1.0,
            step=0.01,
            description='{}:'.format(text),
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
        ))

    rgb_box = widgets.VBox(sliders)
    return sliders, rgb_box


class PixelViewer:
    def __init__(self, rgb_widgets, rgb_box):
        self.ax = None
        self.canvas = None
        self.rgb_widgets = rgb_widgets
        self.rgb_box = rgb_box

        # Assign pixel-viewer to sliders events
        for val in self.rgb_box.children:
            val.observe(self.show_pixel)

        # Get RGB-values
        self.rgb = [val.value for val in self.rgb_widgets]

        # Make figure
        self.fig = plt.figure(self._title_text())
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Show now
        self.show_pixel()

    def _title_text(self):
        return "RGB = {}".format(*self.rgb)

    def _update_pixel_cubes(self):
        # Clear for new plot
        self.ax.cla()
        self.canvas.set_window_title(self._title_text())

        # Get camera angle if there is one
        azim = self.ax.azim
        elev = self.ax.elev

        # Visualize pixel
        output = plot_single_pixel(self.rgb, self.ax)
        self.ax = output[0]

        # Set angle
        self.ax.view_init(elev=elev, azim=azim)

    def show_pixel(self, _=None):
        # Update RGB-values
        self.rgb = [val.value for val in self.rgb_widgets]

        # Update pixel-cubes
        self._update_pixel_cubes()

        # Update angle
        plt.draw()


if __name__ == "__main__":
    out = plot_color_scales()
