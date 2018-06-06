import numpy as np
import pyaudio
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from scipy import signal

def run_livepred(predictor):
        
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication([])

    livepred = LivePredictions(predictor=predictor)
    livepred.show()
    app.exec_()
    livepred.close_stream()


class LivePredictions(QtGui.QMainWindow):
    def __init__(self, predictor, parent=None):
        
        super(LivePredictions, self).__init__(parent)
        
        ### Predictor model ###
        self.predictor = predictor
        self.n_classes = predictor.n_classes
        
        ### Settings ###
        self.rate = 22050    # sampling rate
        self.chunk = 2450    # reading chunk sizes, make it a divisor of sampling rate
        self.nperseg = 490   # samples pr segment for spectrogram, scipy default is 256
        self.noverlap = 0    # overlap between spectrogram windows, scipt default is nperseg // 8 
        self.tapeLength = 4  # length of running tape
        self.start_tape()    # initialize the tape
        self.eps = np.finfo(float).eps
        self.fullTape = False

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)
        
        #  line plot
        self.plot = self.canvas.addPlot()
        self.p1 = self.plot.plot(pen='r')
        self.plot.setXRange(0, self.rate * self.tapeLength)
        self.plot.setYRange(-0.5, 0.5)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')

        self.canvas.nextRow()
        
        # spectrogram
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
        self.img.setLevels([-15, -5])
        
        self.canvas.nextRow()
        
        # create bar chart
        xdict = {0:'a', 1:'b'}
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([xdict.items()])
        
        self.view2 = self.canvas.addViewBox()
        # dummy data
        x = np.arange(self.n_classes)
        y1 = np.linspace(0, self.n_classes, num=self.n_classes)
        self.bg1 = pg.BarGraphItem(x=x, height=y1, width=0.6, brush='r')
        self.view2.addItem(self.bg1)

        #### Start  #####################
        
        self.p = pyaudio.PyAudio()
        self.start_stream()
        self._update()
        
    def start_stream(self):
        self.stream = self.p.open(format=pyaudio.paFloat32, 
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
        data = np.frombuffer(self.raw, dtype=np.float32)
        return self.raw, data
    
    def start_tape(self):
        self.tape = np.zeros(self.tapeLength * self.rate)
        # empty spectrogram tape
        self.f, self.t, self.Sxx = signal.spectrogram(self.tape, 
                                                      self.rate, 
                                                      nperseg=self.nperseg,
                                                      noverlap=self.noverlap)
        self.spec = np.zeros(self.Sxx.shape)
        self.pred = np.zeros((self.n_classes, self.tapeLength * self.rate))
        
    def tape_add(self):
        raw, audio = self.read_stream()
        self.tape[:-self.chunk] = self.tape[self.chunk:]
        self.tape[-self.chunk:] = audio
        
        # spectrogram on whole tape
        # self.f, self.t, self.Sxx = signal.spectrogram(self.tape, self.rate)
        # self.spec = self.Sxx
        
        # spectrogram on last added part of tape
        self.f, self.t, self.Sxx = signal.spectrogram(self.tape[-self.chunk:], 
                                                      self.rate, 
                                                      nperseg=self.nperseg,
                                                      noverlap=self.noverlap)
        self.spec[:, :-len(self.t)] = self.spec[:, len(self.t):]
        self.spec[:, -len(self.t):] = self.Sxx
        

        if self.fullTape:
            # predictions on full tape
            pred = self.predictor.predict(sound_clip=self.tape)[0]
        else:
            # prediction, on some snip of the last part of the signal
            # 1 s seems to be the shortest time frame with reliable predictions
            pred_length = 2
            pred = self.predictor.predict(sound_clip=self.tape[-self.rate * pred_length:])[0]
        self.pred[:, :-self.chunk] = self.pred[:, self.chunk:]
        self.pred[:, -self.chunk:] = np.asarray((self.chunk) * [pred]).transpose()

        
    def _update(self):
        
        try:

            self.tape_add()
            
            psd = abs(self.spec)
            # convert to dB scale
            psd = np.log10(psd + self.eps)

            # self.img.setImage(self.spec.T)
            self.img.setImage(psd.T, autoLevels=False)
            self.p1.setData(self.tape)
            self.bg1.setOpts(height=self.pred[:, -1])

            # self.label.setText('Class: {0:0.3f}'.format(self.pred[-1]))
            
            QtCore.QTimer.singleShot(1, self._update)

            
        except KeyboardInterrupt:
            self.close_stream()