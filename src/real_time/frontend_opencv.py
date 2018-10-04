""" assuming that you are located in the project root when you run this file from the command line"""
if __name__ == "__main__":
    exec(open("notebooks/global_setup.py").read())

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from time import time
from datetime import datetime

class OpenCVFrontendController:
    def __init__(self, interface, 
                 title = None,
                 regime = None,
                 show_crosshair = False,
                 show_labels = False):

        # Global class attributes
        self.interface = interface
        self.show_crosshair = show_crosshair
        self.show_labels = show_labels
        self.title = title
        self.regime = regime #[]
        # Crosshair parameters
        self.crosshair = {}
        self.crosshair["size"] = (224, 224)
        self.crosshair["color"] = (0, 0, 255) #red
        self.crosshair["thickness"] = 3
        # Local class attributes
        self.current_label = None
        self.current_frame = None
        self.frame_size = (1,1)
        self._start_time = None
        self._current_loop_nr = None
        # Regime-specific attributes
        self.photos = {}
        self.photos["pictures"] = []
        self.photos["info"] = []
        
    def run(self):
        self._current_loop_nr = 0
        self.stop_now = False
        self._start_time = time()
        self.interface.loop_initialize()

        if self.show_crosshair == True:
            # Calculating the top-left and bottom-right angles of the rectangle
            crosshair_point_1 = (int(self.frame_size[1] / 2 - self.crosshair["size"][1] / 2), int(self.frame_size[0] / 2 - self.crosshair["size"][0] / 2))
            crosshair_point_2 = (int(self.frame_size[1] / 2 + self.crosshair["size"][1] / 2), int(self.frame_size[0] / 2 + self.crosshair["size"][0] / 2))

        while self.stop_now == False:
            # Run loop step
            self.interface.loop_step()

             # Add crosshair
            if self.show_crosshair == True:
                cv2.rectangle(self.current_frame, crosshair_point_1, crosshair_point_2, self.crosshair["color"], self.crosshair["thickness"], cv2.LINE_8)
            
            # Add labels
            if self.show_labels == True:
                
                # Choose a font and get the text size
                # This is done because CV2 devault putText function does not work well with unicode
                font = ImageFont.truetype("src/utility/arial.ttf", 24)
                text_size = font.getsize(self.current_label)

                ## Calculating the coordinates to put text on the screen
                text_bg_point_1 = (self.frame_size[1] - text_size[0] - 10,0)
                text_bg_point_2 = (self.frame_size[1], text_size[1] + 15)
                ## Drawing the text background
                cv2.rectangle(self.current_frame, text_bg_point_1, text_bg_point_2, (255,255,255), -1)

                # Draw the text
                ## Convert to PIL Image
                cv2_im_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im_rgb)

                draw = ImageDraw.Draw(pil_im)

                ## Draw the text
                draw.text((self.frame_size[1] - text_size[0] - 5, 5), self.current_label, font=font, fill=(0,0,0,255))

                ## Save the image
                self.current_frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
                
            # Show image, flipped on horizontal axis
            cv2.imshow(self.title, self.current_frame)

            # Wait the appropriate time
            key = cv2.waitKey(self.interface.loop_time_milliseconds)
            
            if self.regime == "picture_taking" and key == 13:
                c_time = datetime.now()
                self.photos["pictures"].append(self.current_frame)
                self.photos["info"].append((self._current_loop_nr, str(c_time.date()), str(c_time.time())))
            
            # Check for the end
            if self.interface.loop_stop_check() or self.stop_now or key == 27 or cv2.getWindowProperty(self.title, 0) == -1:
                #print("stopcondition {}, {}, {}".format(self.interface_loop_stop_check(), self.stop_now, cv2.getWindowProperty(self.title, 0)))
                self.interface.loop_finalize()
                cv2.destroyAllWindows()
                self.stop_now = True 

            self._current_loop_nr += 1

    @property
    def current_loop_nr(self) -> int:
        return self._current_loop_nr

    @property
    def start_time(self) -> float:
        return self._start_time