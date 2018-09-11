import cv2

class OpenCVVideoEffects():
    def __init__(self, opencv_frame = None, opencv_text = "", crosshair = {}, frame_size = None, crosshair_size = None):
        
        """
        Adds the overlay effects to the video: captions for the detected objects and crosshair.
        :param opencv_frame: current webcamera frame from the stream
        :param str opencv_text: text to put on the frame (object label)
        :param dict crosshair: crosshair parameters [color, thickness]
        :param tuple frame_size: size of the frame
        :param tuple crosshair_size: size of the crosshair
        """
        
        self.opencv_frame = opencv_frame
        self.opencv_text = opencv_text
        self.crosshair = crosshair
        self.frame_size = frame_size
        self.crosshair_size = crosshair_size

    def update(self):
        # Calculating the top-left and bottom-right angles of the rectangle
        crosshair_point_1 = (int(self.frame_size[1] / 2 - self.crosshair_size[1] / 2), int(self.frame_size[0] / 2 - self.crosshair_size[0] / 2))
        crosshair_point_2 = (int(self.frame_size[1] / 2 + self.crosshair_size[1] / 2), int(self.frame_size[0] / 2 + self.crosshair_size[0] / 2))
        # Getting the text size on the screen to align it nicely
        text_size, _ = cv2.getTextSize(self.opencv_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        # Calculating the coordinates to put text on the screen
        text_bg_point_1 = (self.frame_size[1] - text_size[0] - 10,0)
        text_bg_point_2 = (self.frame_size[1], text_size[1] + 15)
        # Drawing the text background
        cv2.rectangle(self.opencv_frame, text_bg_point_1, text_bg_point_2, (255,255,255), -1)
        # Displaying the text on the frame
        cv2.putText(self.opencv_frame, self.opencv_text,(self.frame_size[1] - text_size[0] - 5, text_size[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0))
        # Drawing the crosshair
        cv2.rectangle(self.opencv_frame, crosshair_point_1, crosshair_point_2, self.crosshair["color"], self.crosshair["thickness"], cv2.LINE_8)

    def setText(self, opencv_text):
        self.opencv_text = opencv_text

    def setFrame(self, opencv_frame):
        self.opencv_frame = opencv_frame

    def setCrossHair(self):
        self.crosshair["color"] = (0, 0, 255) #red
        self.crosshair["thickness"] = 3
 