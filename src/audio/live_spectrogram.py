import numpy as np
import pyaudio
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from scipy import signal

def run_livespec():
        
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication([])

    livespec = LiveSpectrogram()
    livespec.show()
    app.exec_()
    livespec.close_stream()


class LiveSpectrogram(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(LiveSpectrogram, self).__init__(parent)
        
        ### Settings ###
        self.rate = 16000   # sampling rate
        self.chunk = 2**10  # reading chunk sizes
        self.noverlap = 0   # 0 overlap is needed when calculating 
                            # spectrogram on the last added part of the audio signal
        self.tapeLength = 4 # length of running tape
        self.start_tape()   # initialize the tape
        self.eps = np.finfo(float).eps

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)
        
        #  line plot: wave form
        self.lineplot = self.canvas.addPlot()
        self.h2 = self.lineplot.plot(pen='r')
        self.lineplot.setXRange(0, self.rate * self.tapeLength)
        self.lineplot.setYRange(-2**13, 2**13)
        self.lineplot.hideAxis('left')
        self.lineplot.hideAxis('bottom')

        self.canvas.nextRow()
        
        # wiew: spectrogram
        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(False)
        self.view.setRange(QtCore.QRectF(0,0, self.spec.shape[1], 100))
        #  image plot
        self.img = pg.ImageItem() #(border='w')

        self.view.addItem(self.img)
        
        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-100,100])

        #### Start  #####################
        
        self.p = pyaudio.PyAudio()
        self.start_stream()
        self._update()
        
    def start_stream(self):
        self.stream = self.p.open(format=pyaudio.paInt16, 
                                  channels=1, 
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        
    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
    def read_stream(self):
        self.raw = self.stream.read(self.chunk, exception_on_overflow=False)
        data = np.frombuffer(self.raw, dtype=np.int16)
        return self.raw, data
    
    def start_tape(self):
        self.tape = np.zeros(self.tapeLength * self.rate)
        # empty spectrogram tape
        self.f, self.t, self.Sxx = signal.spectrogram(self.tape, 
                                                      self.rate,
                                                      noverlap=self.noverlap)
        self.spec = np.zeros(self.Sxx.shape)
        
    def tape_add(self):
        raw, audio = self.read_stream()
        self.tape[:-self.chunk] = self.tape[self.chunk:]
        self.tape[-self.chunk:] = audio
        
        # spectrogram on last added part of tape
        self.f, self.t, self.Sxx = signal.spectrogram(self.tape[-self.chunk:], 
                                                      self.rate,
                                                      noverlap=self.noverlap)
        self.spec[:, :-len(self.t)] = self.spec[:, len(self.t):]
        self.spec[:, -len(self.t):] = self.Sxx
        
    def _update(self):
        
        try:

            self.tape_add()
            
            psd = abs(self.spec)
            # convert to dB scale
            psd = 20 * np.log10(psd + self.eps)

            # self.img.setImage(self.spec.T)
            self.img.setImage(psd.T, autoLevels=False)
            self.h2.setData(self.tape)

            self.label.setText('wave and spectrogram')
            
            QtCore.QTimer.singleShot(1, self._update)

            
        except KeyboardInterrupt:
            self.close_stream()



if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    # app = QtGui.QApplication(sys.argv)

    livespec = LiveSpectrogram()
    livespec.show()
    app.exec_()
    livespec.close_stream()
