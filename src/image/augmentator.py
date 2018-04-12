from capture_webcam import get_capturer, get_photo_on_keypress
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def _toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and _toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        _toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not _toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        _toggle_selector.RS.set_active(True)

class Augmentator(object):
    def __init__(self):
        self.in_size = None
        self.img = None
        self.crop = None
        self.sbox = None
        self.ebox = None
        self.center = None

    def line_select_callback(self, eclick, erelease):
        '''
        :param eclick:
        :param erelease:
        :return:
        '''
        self.sbox = (int(np.round(eclick.xdata)), int(np.round(eclick.ydata)))
        self.ebox = (int(np.round(erelease.xdata)), int(np.round(erelease.ydata)))

    def mark_object(self, img):
        '''
        :param img:
        :return:
        '''
        self.img = img
        self.in_size = img.shape
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img)
        ax.axis('off')
        _toggle_selector.RS = RectangleSelector(ax, self.line_select_callback,
                                                drawtype='box', useblit=True,
                                                button=[1, 3],  # don't use middle button
                                                minspanx=5, minspany=5,
                                                spancoords='pixels',
                                                interactive=True)
        plt.connect('key_press_event', _toggle_selector)
        ax.set_title('Use mouse to mark object, press q when done', fontsize=30)
        plt.show()

        self.crop = self.img[self.sbox[1]:self.ebox[1], self.sbox[0]:self.ebox[0]]
        self.center = ((self.ebox[0] + self.sbox[0]) / 2.0, (self.ebox[1] + self.sbox[1]) / 2.0)
        return self.crop

    def transform(self, n_trans=3, out_size=(100, 100)):
        '''
        :param n_trans:
        :param out_size:
        :return:
        '''
        if self.crop is None:
            raise ValueError('call Augmentator.mark_object to initialize object to transform')
        trans = [ ]
        for i in range(n_trans):
            M = cv2.getRotationMatrix2D((self.center[0], self.center[1]), 20 * np.random.randn(), 1)
            transformed = cv2.warpAffine(self.img, M, (self.in_size[1], self.in_size[0]))

            rand_l = np.random.randint(0, self.sbox[0])
            rand_r = np.random.randint(0, self.in_size[1] - self.ebox[0])
            rand_d = np.random.randint(0, self.in_size[0] - self.ebox[1])
            rand_u = np.random.randint(0, self.sbox[1])

            cutout = transformed[(self.sbox[1]-rand_u):(self.ebox[1]+rand_d),
                                 (self.sbox[0]-rand_l):(self.ebox[0]+rand_r)]
            transformed = cv2.resize(cutout, dsize=out_size)
            trans.append(transformed)

        return trans

if __name__ == '__main__':
    n_trans = 12
    cam = get_capturer()
    img = get_photo_on_keypress(cam)
    a = Augmentator()
    cut = a.mark_object(img)
    trans = a.transform(n_trans=n_trans, out_size=(300, 300))

    c = np.int(np.ceil(np.sqrt(n_trans)))
    r = np.int(np.ceil(n_trans / c))
    print(r,c)
    fig, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            if i*c+j < n_trans:
                ax[i,j].imshow(trans[i*c+j])
                ax[i,j].axis('off')
            else:
                ax[i,j].axis('off')

    plt.show()
    cam.release()
