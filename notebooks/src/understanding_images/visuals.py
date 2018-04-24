import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Layout, Button
from ipywidgets.widgets import VBox, HBox, FloatSlider, Dropdown, RadioButtons, ToggleButton
from matplotlib.colors import to_rgb

from notebooks.src.understanding_images import d3
from notebooks.src.understanding_images.make_pixel_art import storage_dir

importlib.reload(d3)


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


class ArtViewer:
    def __init__(self, fig_size=(10, 6)):
        self.fig_size = fig_size
        files = list(sorted(list(storage_dir.glob("*.npy"))))
        file_names = [val.name for val in files]

        self.start_button = Button(
            value=False,
            description='Show!',
            disabled=False,
            button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
            icon=''
        )

        self.w_dropdown = Dropdown(
            options=file_names,
            value=file_names[0],
            description='File:',
            disabled=False,
            layout={"margin": "10px"}
        )

        style = {'description_width': 'initial'}
        self.w_camera_position = RadioButtons(
            options=['xyz', 'x', 'y', "z"],
            description='Camera viewpoint:',
            disabled=False,
            style=style,
        )

        self.w_show_mean = ToggleButton(
            value=False,
            description='Show Means/Averages',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            icon='check',
            layout={"justify_content": "space-around", 'width': '250px', "margin": "10px"}
        )

        self.fig = None
        self.ax = None

        self.container = VBox(
            (
                HBox([self.w_dropdown, self.start_button]),
                HBox([self.w_show_mean, self.w_camera_position],
                     layout=Layout(justify_content="space-around", width="100%"),
                    ),
            ),
        )

        # Assign viewer to events
        self.start_button.on_click(self.update)

    def start(self):
        return self.container

    def update(self, _=None):
        self.start_button.disabled = True

        # Get settings
        name = self.w_dropdown.value
        show_means = self.w_show_mean.value
        camera_position = self.w_camera_position.value

        # Make figure
        if self.fig is None:
            self.fig = plt.figure(figsize=self.fig_size)
            self.ax = self.fig.add_subplot(111, projection='3d')

        # Clear axes
        self.ax.cla()

        # Open image
        rgb_image = np.load(str(Path(storage_dir, name)))

        # Plot image
        d3.pixels_image_3d(
            rgb_image=rgb_image,
            no_axis=True,
            insides="full",
            show_means=show_means,
            camera_position=camera_position,
            ax=self.ax
        )

        # Show and enable button
        plt.show()
        self.start_button.disabled = False


class PixelViewer:
    @staticmethod
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

    def __init__(self, view_pixel_cubes=True, view_coordinates=True, fig_size=(12, 8),
                 coordinates_history=False):
        self._fig_size = fig_size
        self._view_pixel_cubes = view_pixel_cubes
        self._view_coordinates = view_coordinates
        self._coordinates_history = coordinates_history

        # Fields
        self.fig = self.canvas = self.axes = None

        # Make widgets
        rgb_widgets = []
        for text in ["Red", "Green", "Blue"]:
            rgb_widgets.append(FloatSlider(
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
        self.rgb_widgets = rgb_widgets

        # Make widget box
        self.rgb_box = VBox(rgb_widgets)

        # Assign pixel-viewer to sliders events
        for val in self.rgb_box.children:
            val.observe(self.show_pixel)

        # Get RGB-values
        self.rgb = [val.value for val in self.rgb_widgets]

        # Start when widgets are displayed
        self.rgb_box.on_displayed(self.show_pixel)

    def start(self):
        return self.rgb_box

    def _title_text(self):
        return "RGB = {}".format(self.rgb)

    def _init_figure(self):
        # Make figure
        self.fig = plt.figure(self._title_text(), figsize=self._fig_size)
        self.canvas = self.fig.canvas

        # Number of axes
        n_axes = sum([self._view_pixel_cubes, self._view_coordinates])

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

    def _update_coordinates(self):
        # Get axes
        axes = self.axes.get("coordinate_axes", None)

        # Reset plot
        if not self._coordinates_history:
            axes.cla()

        # Plot RGB
        axes.plot(
            [0, self.rgb[0] * 0.95],
            [0, self.rgb[1] * 0.95],
            [0, self.rgb[2] * 0.95],
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
        output = PixelViewer.plot_single_pixel(self.rgb, cubes_axes)
        cubes_axes = output[0]

        # Set angle
        cubes_axes.view_init(elev=elev, azim=azim)

        # Save axes
        self.axes["cubes_axes"] = cubes_axes

    def show_pixel(self, _=None):
        # Check if figure is initialized
        if self.fig is None:
            self._init_figure()

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
