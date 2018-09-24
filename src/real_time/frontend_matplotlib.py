""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from matplotlib import pyplot as plt, animation, patches
from _tkinter import TclError
from time import time

class MatplotlibFrontendController:
    def __init__(self, interface, 
                 title = None,
                 show_crosshair = True,
                 show_labels = True):

        # Global class attributes
        self.interface = interface
        self.show_crosshair = show_crosshair
        self.show_labels = show_labels
        self.title = title

        # Crosshair parameters
        self.crosshair = {}
        self.crosshair["size"] = (224, 224)

        self.crosshair["linewidth"] = 1
        self.crosshair["edgecolor"] = "r" # Red

        # Local class attributes
        self.current_label = None
        self.current_label_probability = None
        self.current_frame = None
        self.frame_size = (1,1)
        self._current_loop_nr = 0
        self._start_time = None

        # Settings
        self.block = True
        self.fig = None
        self.artists = []
        self.canvas = None
        self.ax = None
        self._animation = None


        
    def run(self):
        self._current_loop_nr = 0
        self.stop_now = False
        self._start_time = time()
        self.interface.loop_initialize()

        # Default figure
        if self.fig is None:
            self.fig = plt.figure()

        # Set canvas
        self.canvas = self.fig.canvas
        plt.title(self.title)
        self.canvas.set_window_title(self.title)

        # Set close-event
        def closer(_):
            self.stop_now = True
        self.canvas.mpl_connect('close_event', closer)

        # Do animation
        self._animation = animation.FuncAnimation(
            fig = self.fig,
            func = self.__animate_step,
            init_func = self.__initialize_animation,
            interval = self.interface.loop_time_milliseconds,
            repeat = False,
            frames = None,
            blit = False
        )

        # Block main thread
        if self.block:
            self.__wait_for_end()
            plt.close(self.fig)

    def __initialize_animation(self):
        # Set time of start
        self._start_time = time()

        # Initialize parts through interfaces
        self.interface.loop_initialize()

        # Get and set axes
        self.ax = plt.gca() if self.ax is None else self.ax
        plt.sca(self.ax)

        # Add thread-stopper to closing event
        # def closer(_):
        #     self.camera_stream.stop()

        # self.real_time_backend.canvas.mpl_connect('close_event', closer)

        # Title and axis settings
        self.ax.set_title(self.title)

        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        # Add crosshair
        if self.show_crosshair == True:
            # Calculating the top-left and bottom-right angles of the rectangle
            crosshair_xy = (int(self.frame_size[1] / 2 - self.crosshair["size"][1] / 2), int(self.frame_size[0] / 2 - self.crosshair["size"][0] / 2))
            # Drawing a rectangle
            ch = patches.Rectangle(
                xy = crosshair_xy,
                width = self.crosshair["size"][0],
                height = self.crosshair["size"][1],
                fill = False,
                edgecolor = self.crosshair["edgecolor"],
                linewidth = self.crosshair["linewidth"]
            )
            # Add to axes
            self.ax.add_patch(ch)
            # Append to artists
            self.artists.append(ch)

        # Add labels
        if self.show_labels == True:
            self.text = self.fig.text(
                x = 1.0,
                y = 1.0,
                s = "",
                horizontalalignment = "right",
                verticalalignment = "top",
                transform = plt.gca().transAxes,
                backgroundcolor = "darkblue",
                color = "white"
            )
            self.artists.append(self.text)

        return self.artists

    def __animate_step(self, i):
        self._current_loop_nr = i
        # Run animation step
        self.interface.loop_step()
        # OpenCV loads image in BGR format instead of RGB. Reverse the last dimension.
        self.current_frame = self.current_frame[:, :, ::-1]
        # Make image plot
        self._image_plot = plt.imshow(self.current_frame)
        self.artists.append(self._image_plot)
        # Update image plot
        self._image_plot.set_data(self.current_frame)
        # Set text
        if self.show_labels == True:
            self.text.set_text("{}: {:0.4f}".format(self.current_label, self.current_label_probability))
        # Check for end
        if self.interface.loop_stop_check() or self.stop_now:
            self.fig.canvas.stop_event_loop()
            self.__finalize()
            self.stop_now = True

        return self.artists

    def __finalize(self):
        self.interface.finalize()
        plt.close("all")

    def __wait_for_end(self):
        try:
            while not self.stop_now:
                plt.pause(0.2)
        except (TclError, KeyboardInterrupt):
            plt.close("all")
            self.interface.interrupt_handler()

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
