import cv2
import numpy as np
from capture_webcam import get_capturer, get_photo_on_keypress
import matplotlib.pyplot as plt


class Augmentator(object):
    def __init__(self, n_trans=3, out_size=(100, 100)):
        """
        :param n_trans: number of augmentations
        :param out_size: height and width of augmented images
        """
        self.n_trans = n_trans
        self.out_size = out_size
        self.in_size = None
        self.drawing = False
        self.img = None
        self.crop = None
        self.sbox = None
        self.ebox = None
        self.center = None
        self.event = None

    def mark_object(self, img):
        self.img = img[:, :, ::-1]
        self.in_size = img.shape
        while True:
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', a.on_mouse, 0)
            cv2.imshow('image', self.img)
            if a.event == 4:
                cv2.destroyAllWindows()
                break
            cv2.waitKey(1)
        return self.crop

    def on_mouse(self, event, x, y, flags, params):
        global ix, iy
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sbox = [x, y]
            ix, iy = x, y
            self.drawing = True
            self.event = cv2.EVENT_LBUTTONDOWN

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_clone = self.img.copy()
                cv2.rectangle(img_clone, (ix, iy), (x, y), (0, 255, 0), 1)
                cv2.imshow('image', img_clone)
            self.event = cv2.EVENT_MOUSEMOVE
        elif event == cv2.EVENT_LBUTTONUP:
            if x > self.in_size[1] or y > self.in_size[0] or x < 0 or y < 0:
                raise ValueError('Box outsize image, redo')
            self.drawing = False
            self.ebox = [x, y]
            crop = self.img[self.sbox[1]:self.ebox[1], self.sbox[0]:self.ebox[0]]
            self.crop = crop[:, :, ::-1]
            self.center = ((self.ebox[0]+self.sbox[0])/2.0, (self.ebox[1]+self.sbox[1])/2.0)
            self.event = cv2.EVENT_LBUTTONUP

    def transform(self):
        if self.crop is None:
            raise ValueError('call Augmentator.mark_object to initialize object to transform')
        trans = [ ]
        for i in range(self.n_trans):
            M = cv2.getRotationMatrix2D((self.center[0], self.center[1]), 20 * np.random.randn(), 1)
            transformed = cv2.warpAffine(self.img, M, (self.in_size[1], self.in_size[1]))

            move_l = np.min([self.in_size[1]-self.sbox[0], (self.ebox[0]-self.sbox[0])/2.0])
            move_r = np.min([self.in_size[1]-self.ebox[0], (self.ebox[0]-self.sbox[0])/2.0])
            move_d = np.min([self.in_size[0]-self.sbox[1], (self.ebox[1]-self.sbox[1])/2.0])
            move_u = np.min([self.in_size[0]-self.ebox[1], (self.ebox[1]-self.sbox[1])/2.0])

            rand_l = np.random.randint(0, move_l)
            rand_r = np.random.randint(0, move_r)
            rand_d = np.random.randint(0, move_d)
            rand_u = np.random.randint(0, move_u)

            cutout = transformed[(self.sbox[1]-rand_d):(self.ebox[1]+rand_u),(self.sbox[0]-rand_l):(self.ebox[0]+rand_r)]
            transformed = cv2.resize(cutout, dsize=self.out_size)
            trans.append(transformed[:, :, ::-1])

        return trans

if __name__ == '__main__':
    n_trans = 8

    cam = get_capturer()
    img = get_photo_on_keypress(cam)
    img = img.astype('uint8')
    a = Augmentator(n_trans=n_trans, out_size=(200, 200))
    crop = a.mark_object(img)
    trans = a.transform()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(crop)
    ax.axis('off')

    fig, ax = plt.subplots(1, n_trans)
    for i in range(n_trans):
        ax[i].imshow(trans[i])
        ax[i].axis('off')
    plt.show()
    cam.release()
