import random
import warnings
from collections import Iterable
from colorsys import rgb_to_hls, hls_to_rgb, rgb_to_hsv, hsv_to_rgb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import to_rgb


def color_angles_image(image_size=(80, 100), colors=None):
    # Default colors
    if colors is None:
        colors = [
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.8],
            [0.0, 0.8, 0.8],
        ]

    # Image center
    image_center = np.array(image_size) / 2

    # Define rows and columns
    rows = np.arange(0, image_size[0]) - image_center[0]
    cols = np.arange(0, image_size[1]) - image_center[1]

    # Make angles
    rows_expanded = np.flipud(np.expand_dims(rows, 1))
    cols_expanded = np.expand_dims(cols, 0)
    angles = np.arctan(rows_expanded / cols_expanded)

    # Add pi to negative angles, to bottom half and remove nan-pixel
    negative_angles = angles < 0
    angles[negative_angles] += np.pi
    angles[(rows_expanded < 0.5) * (cols_expanded < 0.5)] += np.pi
    angles[(rows_expanded < -0.5) * (cols_expanded > 0.5)] += np.pi
    angles[np.isnan(angles)] = 0

    # Increments for each color
    n_colors = len(colors)
    angle_increments = np.pi * 2 / n_colors

    # Make image
    image = np.zeros((*image_size, 3), dtype=np.float32)

    # Go through colors
    for color_nr, color in enumerate(colors):
        image[(angle_increments * color_nr <= angles) *
              (angles < angle_increments * (color_nr + 1)), :] = color

    return image


def plot_cubes(cube_definitions,
               ax=None, linewidths=1.0, edgecolors="k", face_colors=(0, 0, 1),
               set_axis_limits=True, auto_center=True, alpha=None, mark_points=None):
    # Ensure type
    cube_definitions = np.array(cube_definitions).astype(np.float32)

    # Process colors
    face_colors = _fix_color_tensor(
        face_colors=face_colors, alpha=alpha, n_cubes=len(cube_definitions)
    )

    # Displacement vectors
    vectors = cube_definitions[:, 1:, :] - cube_definitions[:, 0:1, :]

    # Infer remaining points
    inferred_points = np.array([
        cube_definitions[:, 0] + vectors[:, 0] + vectors[:, 1],
        cube_definitions[:, 0] + vectors[:, 0] + vectors[:, 2],
        cube_definitions[:, 0] + vectors[:, 0] + vectors[:, 1] + vectors[:, 2],
        cube_definitions[:, 0] + vectors[:, 1] + vectors[:, 2],
    ])
    inferred_points = np.transpose(inferred_points, axes=(1, 0, 2))

    # Collect points
    points = np.concatenate((cube_definitions, inferred_points), axis=1)

    # Optional centering
    if auto_center:
        axes_max = points.max(0).max(0)
        axes_min = points.min(0).min(0)
        center = (axes_max + axes_min) / 2.
        points -= np.expand_dims(np.expand_dims(center, 0), 0)

    # Gather all points
    all_points = np.reshape(points, (-1, 3))

    # Points are (in the cube's own coordinate system):
    # 0: Cube's origin
    # 1: x-direction
    # 2: y-direction
    # 3: z-direction
    # 4: x + y corner: bottom corner opposite of origin
    # 5: x + z corner
    # 6: x + y + z corner: diametrically opposite of origin
    # 7: y + z corner

    # Compute faces
    # Sides are arrange in positive direction of angle from cube's origin
    faces = np.array([
        [points[:, 0, :], points[:, 1, :], points[:, 4, :], points[:, 2, :]],  # Bottom
        [points[:, 3, :], points[:, 5, :], points[:, 6, :], points[:, 7, :]],  # Top
        [points[:, 0, :], points[:, 3, :], points[:, 5, :], points[:, 1, :]],  # Side 1
        [points[:, 1, :], points[:, 5, :], points[:, 6, :], points[:, 4, :]],  # Side 2
        [points[:, 4, :], points[:, 6, :], points[:, 7, :], points[:, 2, :]],  # Side 3
        [points[:, 2, :], points[:, 7, :], points[:, 3, :], points[:, 0, :]],  # Side 4
    ])
    faces = np.transpose(faces, axes=(2, 0, 1, 3))

    # Get axes
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D

    # Plot faces
    face_plots = []
    for cube_faces, color in zip(faces, face_colors):
        face_plot = Poly3DCollection(cube_faces, linewidths=linewidths, edgecolors=edgecolors)
        face_plot.set_facecolor(color)
        face_plots.append(face_plot)
        ax.add_collection3d(face_plot)

    # Plot the points themselves to force the scaling of the axes
    if set_axis_limits:
        global_max = all_points.max()
        global_min = all_points.min()
        ax.set_xlim([global_min, global_max])
        ax.set_ylim([global_min, global_max])
        ax.set_zlim([global_min, global_max])

    # Check if points are wanted
    if mark_points is not None:
        points_plot = None
        if isinstance(mark_points, int):
            points_plot = points[:, mark_points:mark_points + 1, :]
        elif isinstance(mark_points, Iterable):
            points_plot = points[:, list(mark_points), :]

        if points_plot is not None:
            points_plot = np.reshape(np.transpose(points_plot, (2, 0, 1)), (3, -1))
            points_plot = list(zip(points_plot))
            ax.scatter(*points_plot)

    ax.set_aspect('equal')


def _fix_cube_colors(color=(0.5, 0.5, 0.5)):
    if isinstance(color, str):
        color = to_rgb(color)

    # Convert
    face_colors = np.array(color)

    # Check if a single colors was given for all faces
    if len(face_colors.shape) == 1:
        face_colors = np.expand_dims(face_colors, 0)

    # Ensure color for each side
    if face_colors.shape[0] < 6:
        face_colors = list(face_colors)
        while len(face_colors) < 6:
            face_colors.append(face_colors[-1])
        face_colors = np.array(face_colors)

    return face_colors


def _fix_color_tensor(face_colors=(0.5, 0.5, 0.5), alpha=None, n_cubes=1):
    if isinstance(face_colors, str):
        face_colors = to_rgb(face_colors)

    # Check if a single color was given
    if not isinstance(face_colors[0], Iterable):
        face_colors = [face_colors]

    # Fix each cube
    face_colors = [_fix_cube_colors(val) for val in face_colors]

    # Ensure enough colors
    face_colors = list(face_colors)
    while len(face_colors) < n_cubes:
        face_colors.append(face_colors[-1])

    # Convert to array
    face_colors = np.array(face_colors)

    # Ensure channels (mostly alpha)
    while face_colors.shape[2] < 4:
        face_colors = np.concatenate(
            (face_colors, np.ones((*face_colors.shape[:-1], 1))), axis=len(face_colors.shape[:-1])
        )

    # Check for alpha overwrite
    if isinstance(alpha, (int, float)):
        face_colors[:, :, 3] = alpha

    return face_colors


# def rgb_to_box_definitions(r, g, b):
#
#
#

def rgb_to_cmyk(colors):
    # Ensure format and dimensions
    colors = np.array(colors)
    single_color = False
    if len(colors.shape) == 1:
        colors = np.expand_dims(colors, 0)
        single_color = True
    colors = colors[:, :3]

    # Determine black key
    k = 1 - colors.max(1)

    # Colors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = (1 - colors[:, 0] - k) / (1 - k)
        m = (1 - colors[:, 1] - k) / (1 - k)
        y = (1 - colors[:, 2] - k) / (1 - k)

    # Fix nans (where black)
    c[np.isnan(c)] = 0
    m[np.isnan(m)] = 0
    y[np.isnan(y)] = 0

    # Collect
    new_colors = np.stack([c, m, y, k], 1)

    # If only one color was passed
    if single_color:
        new_colors = np.squeeze(new_colors, 0)

    return new_colors


def cmyk_to_rgb(colors):
    # Ensure format and dimensions
    colors = np.array(colors)
    single_color = False
    if len(colors.shape) == 1:
        colors = np.expand_dims(colors, 0)
        single_color = True

    r = (1 - colors[:, 0]) * (1 - colors[:, 3])
    g = (1 - colors[:, 1]) * (1 - colors[:, 3])
    b = (1 - colors[:, 2]) * (1 - colors[:, 3])

    # Collect
    new_colors = np.stack([r, g, b], 1)

    # If only one color was passed
    if single_color:
        new_colors = np.squeeze(new_colors, 0)

    return new_colors


if __name__ == "__main__":
    with Path("notebooks", "src", "mario_art").open("r") as file:
        lines = file.readlines()
    lines = [val.strip() for val in lines]
    #
    # lines = [
    #     "r,g,b,k,x,y"
    # ]

    plt.close("all")
    #
    # # plt.imshow(color_angles_image())
    #
    move_x = np.array([[1, 0, 0]])
    move_y = np.array([[0, 1, 0]])
    move_z = np.array([[0, 0, 1]])
    base_cube = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

    color_map = dict(
        r=(1, 0, 0),
        w=(1, 1, 1),
        b=(0, 0, 1),
        g=(0, 1, 0),
        x=np.array((139, 69, 19)) / 255,
        y=np.array((255, 224, 189)) / 255,
        i=np.array((255, 255, 0)) / 255,
        k=(0, 0, 0)
    )

    brightness = None
    adjust_white_brightness = False
    rgb_saturation = 1.00

    # Set CMYK intensities based on transparency and wanted ligth through
    cmyk_intensities = np.array([1. / 3, 0.5, 1.0])

    # The designated CMY-component for each RGB-component (arbitrary)
    rgbnr2cmynr = [1, 2, 0]

    # Basic color-components
    see_through = (0.0, 0.0, 0.0, 0.0)
    black = (0.0, 0.0, 0.0, 1.0)

    cubes = []
    colors = []
    for row, line in enumerate(lines):
        for col, char in enumerate(line.split(",")):
            c_cube_base = base_cube + col * move_y + row * move_x

            # Get the wanted color of the pixel
            rgb_color = np.array(color_map[char])
            hls_color = list(rgb_to_hsv(*rgb_color))
            cmyk_color = rgb_to_cmyk(rgb_color)

            # if brightness is not None:
            #     if adjust_white_brightness or rgb_color.sum() < 1 - 1e-5:
            #         hls_color = list(rgb_to_hsv(*rgb_color))
            #         hls_color[2] = hls_color[2] * brightness
            #         rgb_color = np.array(hsv_to_rgb(*hls_color))

            # Get CMYK components
            cmyk_components = np.zeros((3, 4), dtype=np.float32)
            cmyk_components[:, 3] = cmyk_color[3]
            for nr, val in enumerate(cmyk_color[:3]):
                cmyk_components[nr, nr] = val

            # Convert each component to RBG
            subtractive_components = cmyk_to_rgb(cmyk_components)
            # ks = cmyk_components[:, 3]

            # Split into three pixels
            for nr, component_intensity in enumerate(rgb_color):
                # Get current subtractive color and set transparency
                subtractive_color = subtractive_components[rgbnr2cmynr[nr]]
                subtractive_color = np.concatenate((subtractive_color, [cmyk_intensities[nr]]), 0)

                # Make RGB-wrapping
                c_rgb_color = np.array([0., 0., 0., 1.])
                c_rgb_color[nr] = component_intensity

                # Bottom color
                bottom_color = black if nr == 2 else see_through

                # Distribute face colors
                c_face_colors = [
                    bottom_color, subtractive_color,
                    c_rgb_color, c_rgb_color, c_rgb_color, c_rgb_color,
                ]

                # Append cube and colors
                cubes.append(c_cube_base - nr * move_z)
                colors.append(c_face_colors)

    colors = np.array(colors)
    cubes = np.stack(cubes, axis=0)

    # cubes = [
    #     base_cube,
    #     base_cube + move_z,
    #     # base_cube + move_z*2,
    # ]
    # colors = [
    #     (1, 0, 0, 0.2),
    #     (0, 1, 0, 0.8),
    #     (0, 0, 1, 0.4),
    #     (0, 0, 1, 0.4),
    #     (0, 0, 0, 0.3),
    # ]

    # cube_definitions = []
    # cube_definitions.append(cube_definitions[0] + np.expand_dims([1, 0, 0], 0))
    plot_cubes(
        cubes,
        face_colors=colors,
        linewidths=0.0,
        auto_center=True
    )
    ax = plt.gca()
    # ax.set_axis_off()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
