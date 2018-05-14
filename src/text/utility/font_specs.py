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
        # Determine font_size
        text_height, _ = Font._measure_a_text(text="em", fontname=fontname,
                                              fontsize=font_size)
        hf_ratio = text_height / font_size

        # Leave out space for now
        chars = [char for char in Font.measured_characters if char != " "]

        # Go through all measured characters
        char_widths = dict()
        measurements = []
        for char in chars:
            # Measure character
            height, width = Font._measure_a_text(text=char, fontname=fontname, fontsize=font_size)

            # Determine ratio
            measurements.append((char, text_height / width, height / width))
            char_widths[char] = width

            # Get rid of figure
            plt.close("all")

        # Approximate space between characters
        _, width = Font._measure_a_text(text="".join(chars), fontname=fontname, fontsize=font_size)
        est_width = sum([char_widths[char] for char in chars])
        char_sep = (width - est_width) / (len(chars) - 1)

        # Measure space
        _, pre_width = Font._measure_a_text(text="".join(chars), fontname=fontname, fontsize=font_size)
        _, post_width = Font._measure_a_text(text=" ".join(chars), fontname=fontname, fontsize=font_size)
        width = (post_width - pre_width) / (len(chars) - 1)
        measurements.append((" ", text_height / width, 0))

        # Store ratios of characters
        with Path(Font.font_directory, fontname + ".csv").open("w") as file:
            writer = csv.writer(file, **Font.font_csv_settings)
            for measurement in measurements:
                writer.writerow(measurement)

        # Store simple information
        config = ConfigParser()
        config["FONT"] = dict(
            name=fontname,
            height_font_ratio=hf_ratio,
            char_sep=char_sep,
        )
        with Path(Font.font_directory, fontname + ".ini").open("w") as configfile:
            config.write(configfile)

    @staticmethod
    def measure_all_fonts():
        print("Measured:")
        for font in Font.font_families:
            print("\t{}".format(font))
            Font.measure_font(fontname=font)
        plt.close("all")
