import csv
import string
from configparser import ConfigParser
from functools import lru_cache
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class Font:
    font_families = ["serif", "sans-serif", "monospace"]
    font_csv_settings = dict(delimiter=' ', quoting=csv.QUOTE_MINIMAL)

    measured_characters = string.ascii_letters + string.digits + string.punctuation

    text_directory = Path(*Path(__file__).parts[:-list(reversed(Path(__file__).parts)).index("text")])
    font_directory = Path(text_directory, "font_info")
    if not font_directory.exists():
        font_directory.mkdir()

    def __init__(self, fontname):
        self.fontname = fontname
        self.hw_ratios = dict()
        self.hw_char_ratios = dict()
        self.hf_ratio = None

        # Check whether font has been measured
        config_path = Path(Font.font_directory, fontname + ".ini")
        if config_path.exists():

            # Get simple information
            config = ConfigParser()
            config.read(str(config_path))
            self.hf_ratio = float(config["FONT"]["height_font_ratio"])

            # Get characterwise information
            with Path(Font.font_directory, fontname + ".csv").open("r") as file:
                reader = csv.reader(file, **Font.font_csv_settings)
                for key, *val in reader:
                    self.hw_ratios[key] = float(val[0])
                    self.hw_char_ratios[key] = float(val[1])

            # Mean
            self.hw_ratio_mean = np.mean(list(self.hw_ratios.values()))

        else:
            raise ValueError("This font does not seem to have been measured: {}".format(fontname))

    @staticmethod
    @lru_cache(maxsize=10)
    def get_font(fontname):
        return Font(fontname=fontname)

    @staticmethod
    def _measure_a_text(text, fontname, fontsize):
        # Make a figure and get renderer
        plt.close("all")
        fig = plt.figure(figsize=(10, 10))
        renderer = fig.canvas.get_renderer()
        ax = plt.gca()
        ax.set_aspect("equal")

        # Write text
        text = plt.text(0.5, 0.5, text, fontname=fontname, fontsize=fontsize)

        # Get bounding box and determine width and height
        bb = text.get_window_extent(renderer=renderer)
        return bb.height, bb.width

    @staticmethod
    def measure_font(fontname="serif", font_size=15):
        """
        Measures the height-width ratio of each character in a font and stores it in the font_info directory.
        :param fontname:
        :param font_size
        """
        # Determine text height
        text_height, text_width = Font._measure_a_text(text=Font.measured_characters, fontname=fontname,
                                                       fontsize=font_size)
        hf_ratio = text_height / font_size

        # Go through all measured characters
        measurements = []
        for char in Font.measured_characters:
            # Measure character
            height, width = Font._measure_a_text(text=char, fontname=fontname, fontsize=font_size)

            # Determine ratio
            measurements.append((char, text_height / width, height / width))

            # Get rid of figure
            plt.close("all")

        # Store ratios of characters
        with Path(Font.font_directory, fontname + ".csv").open("w") as file:
            writer = csv.writer(file, **Font.font_csv_settings)
            for measurement in measurements:
                writer.writerow(measurement)

        # Store simple information
        config = ConfigParser()
        config["FONT"] = dict(
            name=fontname,
            height_font_ratio=hf_ratio
        )
        with Path(Font.font_directory, fontname + ".ini").open("w") as configfile:
            config.write(configfile)

    @staticmethod
    def measure_all_fonts():
        print("Measured:")
        for font in Font.font_families:
            print("\t{}".format(font))
            Font.measure_font(fontname=font)
