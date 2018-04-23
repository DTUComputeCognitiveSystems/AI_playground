import numpy as np


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