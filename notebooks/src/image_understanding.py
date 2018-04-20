from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from notebooks.src.understanding_images.d3 import pixels_3d

if __name__ == "__main__":
    plt.close("all")

    # Get rgb image
    rgb_image = np.load(str(Path("notebooks", "src", "understanding_images", "data", "art1.npz")))

    # Plot 3D pixels
    pixels_3d(
        rgb_image=rgb_image,
        from_top=False
    )
