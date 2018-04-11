import cv2
import numpy as np
import matplotlib.pyplot as plt
from capture_webcam import get_capturer, get_photo_on_keypress

class BackgroundRemover(object):
    def __init__(self, frame, threshold=(25, 255)):
        self.background_frame = frame
        self.threshold = threshold
        self.background = cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2GRAY)
        self.background = cv2.GaussianBlur(self.background, (21, 21), 0)
        
    def mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        difference = cv2.absdiff(gray, self.background)
        thresh = cv2.threshold(difference, self.threshold[0], self.threshold[1], 
                               cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        return thresh
        
    def foreground(self, img):
        n_channel = img.shape[-1]
        mask = self.mask(img) > 0
        mask = np.tile(mask[:,:,np.newaxis], (1,1,n_channel))
        foreground = mask*img
        return foreground 
        
if __name__ == '__main__':
    cam = get_capturer()
    background = get_photo_on_keypress(cam, 'background')
    br = BackgroundRemover(background)
    for i in range(5):
        frame = get_photo_on_keypress(cam, 'frame' + str(i))
        foreground = br.foreground(frame)
        plt.figure(i)
        plt.imshow(foreground)
    
    cam.release()
    