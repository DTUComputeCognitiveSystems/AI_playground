""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

from matplotlib import pyplot as plt, animation
from time import time

class OpenCVFrontendController:
    def __init__(self, interface, 
                 title = None,
                 show_crosshair = False,
                 show_labels = False):

        # Global class attributes
        self.interface = interface
        self.show_crosshair = show_crosshair
        self.show_labels = show_labels
        self.title = title

        # Crosshair parameters
        self.crosshair = {}
        self.crosshair["size"] = (224, 224)
        self.crosshair["color"] = (0, 0, 255) #red
        self.crosshair["thickness"] = 3

        # Local class attributes
        self.current_label = None
        self.current_label_probability = None
        self.current_frame = None
        self.frame_size = (1,1)
        self.start_time = None
        
    def run(self):
        self.current_loop_nr = 0
        self.stop_now = False
        self.start_time = time()
        self.interface.loop_initialize()

        if self.show_crosshair == True:
            # Calculating the top-left and bottom-right angles of the rectangle
            crosshair_point_1 = (int(self.frame_size[1] / 2 - self.crosshair["size"][1] / 2), int(self.frame_size[0] / 2 - self.crosshair["size"][0] / 2))
            crosshair_point_2 = (int(self.frame_size[1] / 2 + self.crosshair["size"][1] / 2), int(self.frame_size[0] / 2 + self.crosshair["size"][0] / 2))

        while self.stop_now == False:
            self.current_loop_nr += 1

            # Run loop step
            self.interface.loop_step()

            # Flip image horizontally
            self.current_frame = cv2.flip(self.current_frame, 1)

             # Add crosshair
            if self.show_crosshair == True:
                cv2.rectangle(self.current_frame, crosshair_point_1, crosshair_point_2, self.crosshair["color"], self.crosshair["thickness"], cv2.LINE_8)
            
            # Add labels
            if self.show_labels == True:
                ## Getting the text size on the screen to align it nicely
                text_size, _ = cv2.getTextSize("{}: {:0.4f}".format(self.current_label, self.current_label_probability), cv2.FONT_HERSHEY_DUPLEX, 1, 1)
                ## Calculating the coordinates to put text on the screen
                text_bg_point_1 = (self.frame_size[1] - text_size[0] - 10,0)
                text_bg_point_2 = (self.frame_size[1], text_size[1] + 15)
                ## Drawing the text background
                cv2.rectangle(self.current_frame, text_bg_point_1, text_bg_point_2, (255,255,255), -1)
                ## Displaying the text on the frame
                cv2.putText(self.current_frame, "{}: {:0.4f}".format(self.current_label, self.current_label_probability),(self.frame_size[1] - text_size[0] - 5, text_size[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
        
            # Show image, flipped on horizontal axis
            cv2.imshow(self.title, self.current_frame)

            # Wait the appropriate time
            key = cv2.waitKey(self.interface.loop_time_milliseconds)
            
            # Check for end
            if self.interface.loop_stop_check() or self.stop_now or key == 27 or cv2.getWindowProperty(self.title, 0) == -1:
                #print("stopcondition {}, {}, {}".format(self.interface_loop_stop_check(), self.stop_now, cv2.getWindowProperty(self.title, 0)))
                self.interface.loop_finalize()
                cv2.destroyAllWindows()
                self.stop_now = True 





from _tkinter import TclError




from src.real_time.base_backend import BackendLoop, BackendInterface


class MatplotlibLoop(BackendLoop):
    def __init__(self, backend_interface=(),
                 title="Real time animation",
                 fig=None, block=False, blit=False):
        """
        :param BackendInterface backend_interface:
        :param str title:
        :param fig:
        :param bool block:
        :param bool blit:
        """
        # Settings
        self.blit = blit
        self.title = title
        self.block = block
        self.fig = fig

        # Connect interface
        if isinstance(backend_interface, Iterable):
            backend_interface = tuple(backend_interface)
        elif not isinstance(backend_interface, tuple):
            backend_interface = (backend_interface,)
        super().__init__(*backend_interface)

        # Fields
        self.artists = []
        self.canvas = self.ax = self._animation = self._current_loop_nr = self._start_time = None

    def _start(self):
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
            fig=self.fig,
            func=self.__animate_step,
            init_func=self.__initialize_animation,
            interval=self.loop_time_milliseconds,
            repeat=False,
            frames=None,
            blit=self.blit
        )

        # Block main thread
        if self.block:
            self.__wait_for_end()
            plt.close(self.fig)

    def __initialize_animation(self):

        # Initialize parts through interfaces
        self.interface_loop_initialization()

        # Make sure figure is drawn
        plt.draw()
        plt.show()

        # Set time of start
        self._start_time = time()

        return self.artists

    def __animate_step(self, i):
        self._current_loop_nr = i

        # Run animation step
        self.interface_loop_step()

        # Check for end
        if self.interface_loop_stop_check() or self.stop_now:
            self.fig.canvas.stop_event_loop()
            self.__finalize()
            self.stop_now = True

        return self.artists

    def __finalize(self):
        self.interface_finalize()
        plt.close("all")

    def __wait_for_end(self):
        try:
            while not self.stop_now:
                plt.pause(0.2)
        except (TclError, KeyboardInterrupt):
            plt.close("all")
            self.interface_interrupt_handler()

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time
