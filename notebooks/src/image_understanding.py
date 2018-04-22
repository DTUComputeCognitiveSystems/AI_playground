from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from matplotlib.colors import to_rgb
import importlib

from notebooks.src.understanding_images import d3
from notebooks.src.understanding_images.make_pixel_art import storage_dir

importlib.reload(d3)


def plot_art(n=1, no_axis=True, fig_size=(12, 8), show_means=0):
    rgb_image = np.load(str(Path(storage_dir, "art{}.npy".format(n))))
    plt.figure(figsize=fig_size)
    d3.pixels_image_3d(
        rgb_image=rgb_image,
        no_axis=no_axis,
        insides="full",
        show_means=show_means,
    )


def plot_color_scales(
        scales="red,lime,blue,yellow,magenta,cyan,white",
        n_shapes=10, x_spread=3, y_spread=0, z_spread=3.5, fig_size=(12, 8)):
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
    plt.figure(figsize=fig_size)
    _ = d3.pixels_3d(
        positions=positions,
        pixel_colors=pixel_colors,
        camera_position="xy",
        no_axis=True,
        linewidths=0.1,
        insides="full",
    )


def plot_single_pixel(color, ax=None, insides="full"):
    color = np.array(to_rgb(color))
    output = d3.pixels_3d(
        positions=[(0, 0, 0)],
        pixel_colors=[color],
        camera_position="x",
        no_axis=True,
        linewidths=0.1,
        ax=ax,
        insides=insides,
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
    def __init__(self, rgb_widgets, rgb_box, view_pixel_cubes=True, view_coordinates=True, fig_size=(12, 8),
                 coordinates_history=False):
        self.rgb_widgets = rgb_widgets
        self.rgb_box = rgb_box
        self._view_pixel_cubes = view_pixel_cubes
        self._view_coordinates = view_coordinates
        self._coordinates_history = coordinates_history

        # Assign pixel-viewer to sliders events
        for val in self.rgb_box.children:
            val.observe(self.show_pixel)

        # Get RGB-values
        self.rgb = [val.value for val in self.rgb_widgets]

        # Make figure
        self.fig = plt.figure(self._title_text(), figsize=fig_size)
        self.canvas = self.fig.canvas

        # Number of axes
        n_axes = sum([view_pixel_cubes, view_coordinates])

        # Prepare axes
        self.axes = dict()
        ax_nr = 1
        if self._view_pixel_cubes:
            self.axes["cubes_axes"] = self.fig.add_subplot(1, n_axes, ax_nr, projection='3d')
            ax_nr += 1
        if self._view_coordinates:
            self.axes["coordinate_axes"] = self.fig.add_subplot(1, n_axes, ax_nr, projection='3d')
            self.axes["coordinate_axes"].view_init(elev=20, azim=35)
            ax_nr += 1

        # Show now
        self.show_pixel()

    def _title_text(self):
        return "RGB = {}".format(self.rgb)

    def _update_coordinates(self):
        # Get axes
        axes = self.axes.get("coordinate_axes", None)

        # Reset plot
        if not self._coordinates_history:
            axes.cla()

        # Plot RGB
        axes.plot(
            [0, self.rgb[0]*0.95],
            [0, self.rgb[1]*0.95],
            [0, self.rgb[2]*0.95],
            c="k",
            zorder=-1,
        )
        axes.scatter(
            *[[val] for val in self.rgb],
            s=100,
            zorder=1,
            c=self.rgb,
            edgecolors="k",
        )

        # Set limits
        axes.set_xlim(0., 1.1)
        axes.set_ylim(0., 1.1)
        axes.set_zlim(0., 1.1)

        # Set labels
        axes.set_xlabel("Red")
        axes.set_ylabel("Green")
        axes.set_zlabel("Blue")

    def _update_pixel_cubes(self):
        # Get axes
        cubes_axes = self.axes.get("cubes_axes", None)

        # Clear for new plot
        cubes_axes.cla()
        self.canvas.set_window_title(self._title_text())

        # Get camera angle if there is one
        azim = cubes_axes.azim
        elev = cubes_axes.elev

        # Visualize pixel
        output = plot_single_pixel(self.rgb, cubes_axes)
        cubes_axes = output[0]

        # Set angle
        cubes_axes.view_init(elev=elev, azim=azim)

        # Save axes
        self.axes["cubes_axes"] = cubes_axes

    def show_pixel(self, _=None):
        # Update RGB-values
        self.rgb = [val.value for val in self.rgb_widgets]

        # Update pixel-cubes
        if self._view_pixel_cubes:
            self._update_pixel_cubes()

        # Update coordinates plot
        if self._view_coordinates:
            self._update_coordinates()

        # Update angle
        plt.draw()


if __name__ == "__main__":
    out = plot_color_scales()
