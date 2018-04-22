from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from notebooks.src.understanding_images.colors import rgb_to_cmyk, cmyk_to_rgb

_default_rgb_to_white = False


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
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')

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
        ax.set_xlim(*[global_min, global_max])
        ax.set_ylim(*[global_min, global_max])
        ax.set_zlim(*[global_min, global_max])

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

    return ax, face_plots,


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


def pixels_image_3d(rgb_image, rgb_to_white=_default_rgb_to_white, no_axis=True, camera_position=None, mask=None,
                    linewidths=0.0, insides="cmyk",
                    show_means=0):
    positions = []
    pixel_colors = []
    new_mask = None if mask is None else []
    for row in range(rgb_image.shape[0]):
        for col in range(rgb_image.shape[1]):
            rgb_color = rgb_image[row, col, :]
            positions.append((row, col))
            pixel_colors.append(rgb_color)

            if mask is not None:
                new_mask.append(mask[row, col])

    if show_means:
        if isinstance(show_means, bool):
            show_means = 3

        # Compute means
        row_mean = rgb_image.mean(0)
        col_mean = rgb_image.mean(1)

        # Set positions
        positions.extend([(rgb_image.shape[0] + show_means, val) for val in range(rgb_image.shape[1])])
        positions.extend([(val, rgb_image.shape[1] + show_means) for val in range(rgb_image.shape[0])])

        # Set colors
        pixel_colors.extend(list(row_mean))
        pixel_colors.extend(list(col_mean))

    pixels_3d(
        positions=positions,
        pixel_colors=pixel_colors,
        rgb_to_white=rgb_to_white,
        no_axis=no_axis,
        camera_position=camera_position,
        mask=new_mask,
        linewidths=linewidths,
        insides=insides,
    )


def pixels_3d(positions, pixel_colors, rgb_to_white=_default_rgb_to_white, no_axis=True, camera_position=None,
              mask=None, linewidths=0.0, ax=None, insides="cmyk"):
    # Cube building blocks
    move_x = np.array([[1, 0, 0]])
    move_y = np.array([[0, 1, 0]])
    move_z = np.array([[0, 0, 1]])
    base_cube = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # Set CMYK intensities based on transparency and wanted light through
    # cmyk_transparancies = np.array([0.5, 0.5, 1.0])
    cmyk_transparancies = np.array([1. / 3, 0.5, 1.0])

    # The designated CMY-component for each RGB-component (pretty arbitrary)
    rgbnr2cmynr = [1, 2, 0]

    # Basic color-components
    see_through = (0.0, 0.0, 0.0, 0.0)
    black = (0.0, 0.0, 0.0, 1.0)

    # Holders
    cubes = []
    cube_colors = []

    # Ensure number of dimensions in positions
    positions = [
        list(val) + [0] * (3 - len(val)) for val in positions
    ]

    # Checks
    do_cmyk = "cmy" in insides

    # Go through all pixels
    for pix_nr, (rgb_color, position) in enumerate(zip(pixel_colors, positions)):

        do_pixel = True
        if mask is not None:
            do_pixel = mask[pix_nr]

        if do_pixel:
            c_cube_base = base_cube + position[0] * move_x + position[1] * move_y + position[2] * move_z

            # Check if CMYK method is used
            subtractive_components = None
            if do_cmyk:
                # Get the wanted color of the pixel
                cmyk_color = rgb_to_cmyk(rgb_color)

                # Get CMYK components
                cmyk_components = np.zeros((3, 4), dtype=np.float32)
                cmyk_components[:, 3] = cmyk_color[3]
                for nr, val in enumerate(cmyk_color[:3]):
                    cmyk_components[nr, nr] = val

                # Convert each component to RBG
                subtractive_components = cmyk_to_rgb(cmyk_components)

            # Split into three pixels
            for nr, component_intensity in enumerate(rgb_color):
                # Make RGB-wrapping
                if rgb_to_white:
                    c_rgb_color = np.ones(4) - component_intensity
                    c_rgb_color[nr] = 1
                else:
                    c_rgb_color = np.zeros(4)
                    c_rgb_color[nr] = component_intensity
                c_rgb_color[3] = 1

                # Bottom color
                bottom_color = black if nr == 2 else see_through

                # Check if CMYK method is used
                if do_cmyk:
                    # Get current subtractive color and set transparency
                    subtractive_color = subtractive_components[rgbnr2cmynr[nr]]
                    subtractive_color = np.concatenate((subtractive_color, [cmyk_transparancies[nr]]), 0)

                    # Check scheme for insides
                    insides_color = subtractive_color

                else:
                    insides_color = np.array(list(rgb_color) + [cmyk_transparancies[nr]])

                # Distribute face colors
                c_face_colors = [
                    bottom_color, insides_color,
                    c_rgb_color, c_rgb_color, c_rgb_color, c_rgb_color,
                ]

                # Append cube and colors
                cubes.append(c_cube_base - nr * move_z)
                cube_colors.append(c_face_colors)

    # Make arrays
    cube_colors = np.array(cube_colors)
    cubes = np.stack(cubes, axis=0)

    # Plot cubes
    output = plot_cubes(
        cubes,
        face_colors=cube_colors,
        auto_center=True,
        linewidths=linewidths,
        ax=ax
    )

    # Axes settings
    ax = output[0]

    if no_axis:
        ax.set_axis_off()

    if camera_position is not None:
        arguments = 0
        axes_angles = np.array([0, 0])
        if "x" in camera_position:
            axes_angles += np.array([0, -90])
            arguments += 1
        if "y" in camera_position:
            axes_angles += np.array([0, 0])
            arguments += 1
        if "z" in camera_position:
            axes_angles += np.array([90, 0])
            arguments += 1

        if arguments != 0:
            ax.view_init(*(axes_angles / arguments))

    return output
