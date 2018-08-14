import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import Layout, FloatText, Box, VBox, FloatSlider, HBox
import pyaudio
import wave
import numpy as np
import time
import pylab
import sys
from scipy import signal
# normalization in an imshow plot to 0-1 range on log scale
from matplotlib.colors import LogNorm  
from IPython import display            

class SoundPlotter:
    def __init__(self, recorder=None, limits=(1, 100)):
        self.limits = limits

        # Default values
        if recorder is None:
            self.rate = 44100              
            self.framesize = 2**13         
            self.seconds = 1
            self.noframes = int((self.rate * self.seconds) / self.framesize)  # number of frames needed
            self.sound = 2**13 * np.random.rand(NOFRAMES * FRAMESIZE) - 2**13/2
        else:
            self.sound = recorder.sound
            self.rate = recorder.rate
        
        # signal length
        self.n = len(self.sound)
        
        # slider names
        names = ['window zoom', 'location']
        
        # Make widgets
        self.widgets = [
            FloatSlider(
                min=limits[0],
                max=limits[1],
                step=0.1,
                description=names[idx],
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            for idx in range(2)
        ]

        # Make widget box
        self.box = HBox(self.widgets)

        # Set fields
        self.fig = self.ax = None

        # Make figure when widgets are shown
        self.box.on_displayed(self.plot_function)
        
        # Make widgets pass data to plotting function
        for widget in self.widgets:
            widget.observe(self.plot_function)

    def plot_function(self, _=None):

        # Initialize figure
        if self.ax is None:
            self.fig = plt.figure()
            self.ax = plt.gca()
        else:
            self.ax.cla()
            
        # Chage slider values to percentages
        val0 = self.widgets[0].value / 100
        val1 = self.widgets[1].value / 100
            
        # Window size, can go from full signal to 2**9
        # if slider is at 1, full signal
        window_size = self.n - int((self.n - 2**7) * val0)
        
        # Window location, depending on slider an window size
        # if slider is at 1, location is far left
        window_location = int(window_size / 2) + int((self.n - window_size) * val1)
            
        # Find relevant zoom
        xmin = window_location - int(window_size / 2)
        xmax = window_location + int(window_size / 2)
        ylim = np.max(np.abs(self.sound[xmin:xmax]))

        # Set axes
        # self.ax.set_xlim(xmin/RATE, xmax/RATE)
        self.ax.set_ylim(-ylim, ylim)

        # Title and labels
        self.ax.set_title("Sound")
        self.ax.set_xlabel('Time [ms]')
        
        # Plot 
        s = self.sound[xmin: xmax]
        plotrange = xmax - xmin
        if plotrange < 2**11:
            # plot with markers
            self.ax.plot(np.linspace(0, (len(s)/self.rate) * 1000, len(s)), s, linestyle='-', marker='o', markersize=2)
        else:
            # plot without markers
            self.ax.plot(np.linspace(0, (len(s)/self.rate) * 1000, len(s)), s)
            
    def start(self):
        return self.box

class miniSound:
    
    def __init__(self, rate=16000, freq1=1024, freq2=None, seconds=1):
        
        self.rate = rate               # sampling rate (Hz)
        self.record_seconds = seconds  # how long should the recording be
        
        # number of time points in the array
        self.n = int(rate * self.record_seconds)
        
        # how much should the array be padded to make it a multiple of the sampling rate?
        self.n_pad = self.n % rate        
        
        # create the sound signal
        t = np.arange(self.n)
        wave_fun = lambda freq: np.sin(2 * np.pi * (freq/rate) * t)
        
        if freq2:
            self.sound = wave_fun(freq1) + wave_fun(freq2)
        else:
            self.sound = wave_fun(freq1) 
        
        # pad the array with zeros if needed
        self.sound = np.hstack((self.sound, np.zeros(self.n_pad)))
        
    def playBack(self):
        ma = np.abs(self.sound).max()
        data = ''
        #generating waves
        for i in range(self.n + self.n_pad):    
            data += chr(int((self.sound[i]/ma) *127+128))  # integers from 0:255 turned into characters

        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(1), 
                        channels=1, 
                        rate=self.rate, 
                        output=True)

            stream.write(data, len(data))
            stream.stop_stream()
            stream.close()
            p.terminate()

        except:
            print("Error:", sys.exc_info()[0])




def SoundPlotter2(miniSound):
    
    max_freq = 2000
    
    # Power spectrum
    f, fft = signal.periodogram(miniSound.sound, miniSound.rate, scaling='spectrum')
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # show 0.01s
    samp = int(miniSound.rate * 0.01)
    ax1.plot(np.linspace(0, samp, samp) / miniSound.rate, miniSound.sound[:samp])
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax2.plot(f[:max_freq], fft[:max_freq])       
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Power')
    # make sure axis labels and titles do not overlap
    plt.tight_layout()


class SoundPlotter3:
    def __init__(self, recorder=None, limits=(0, 100), log_y=False):
        self.limits = limits
        self.log_y = log_y

        # Default values
        if recorder is None:
            self.rate = 44100              # sampling rate (Hz)
            self.seconds = 1
            self.noframes = int((self.rate * self.seconds) / self.framesize)  # number of frames needed
            sound = 2**13 * np.random.rand(NOFRAMES * FRAMESIZE) - 2**13/2
        self.sound = recorder.sound
        self.rate = recorder.rate
        
        # signal length
        self.n = len(self.sound)
        self.midpoint = int(self.n / 2)
        
        # slider names
        names = ['zoom', 'location']
        
        # Make widgets
        self.widgets = [
            FloatSlider(
                min=limits[0],
                max=limits[1],
                step=0.1,
                description=names[idx],
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            for idx in range(2)
        ]


        # Make widget box
        self.box = HBox(self.widgets)

        # Set fields
        self.fig = self.ax1 = None

        # Make figure when widgets are shown
        self.box.on_displayed(self.plot_function)

        
        # Make widgets pass data to plotting function
        for widget in self.widgets:
            widget.observe(self.plot_function)

    def plot_function(self, _=None):

        # Initialize figure
        if self.ax1 is None:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1)
        else:
            self.ax1.cla()
            self.ax2.cla()
            self.ax3.cla()
            # self.cb.remove()
            
        ############################
        # Window size and location #
        ############################
            
        # Change slider values to percentages
        val0 = self.widgets[0].value / 100
        val1 = self.widgets[1].value / 100
            
        # Window size, can go from full signal to 2**9
        # if slider is at 0, full signal
        window_size = self.n - int((self.n - 2**9) * val0)
        
        # Window location, depending on slider an window size
        # if slider is at 0, location is far left
        window_location = int(window_size / 2) + int((self.n - window_size) * val1)
        
        xmin_sample = window_location - int(window_size / 2)
        xmax_sample = window_location + int(window_size / 2)
        
        # from samples to milli seconds
        xmin = (xmin_sample / self.rate) * 1000  
        xmax = (xmax_sample / self.rate) * 1000
        
        ############################
        # Signals                  #
        ############################
        
        s = self.sound
        f, fft = signal.periodogram(self.sound[xmin_sample: xmax_sample], self.rate, scaling='spectrum')
        
        ############################
        # Plot settings            #
        ############################
            
        # Y limit
        ylim = np.max(np.abs(self.sound)) 

        # Set axes
        self.ax1.set_ylim(-ylim, ylim)

        # Title and labels
        self.ax1.set_title("Sound, window size {0:4d} samples".format(window_size))
        self.ax1.set_xlabel('Time [ms]')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_xlim(0, (self.n / self.rate) * 1000)
        
        self.ax2.set_xlabel('Frequency [Hz]')
        self.ax2.set_ylabel('Power')
        # self.ax2.yaxis.set_visible(False)
        self.ax2.set_title('Frequency spectrum of window')
        self.ax2.set_xlim(f[0], f[-1])
        
        self.ax3.set_ylabel('Frequency [Hz]')
        self.ax3.set_xlabel('Time [ms]')
        self.ax3.set_title('Frequency spectrum of window, shown as colors')
        
        # Plot
        self.ax1.plot(np.linspace(0, (self.n / self.rate) * 1000, self.n), s, linestyle='-') 
        self.ax1.plot([xmin, xmin], [-ylim, ylim], 'orange')
        self.ax1.plot([xmax, xmax], [-ylim, ylim], 'orange')
        self.ax1.plot([xmin, xmax], [-ylim, -ylim], 'orange')
        self.ax1.plot([xmin, xmax], [ylim, ylim], 'orange')
        
        if self.log_y:
            self.ax2.semilogy(f, fft)
            
            spec = np.zeros((len(fft), self.n))
            spec[:, xmin_sample:xmin_sample + window_size] = np.asarray([fft] * window_size).transpose()
            extent = (0, (self.n / self.rate) * 1000, f[-1], f[0])
            im = self.ax3.imshow(spec, aspect='auto', extent=extent,
                    cmap = 'jet', norm = LogNorm(vmin=10**-12,vmax=10**-6))
            plt.gca().invert_yaxis()
            
            # make sure axis labels and titles do not overlap
            plt.tight_layout()
            
        else:
            self.ax2.plot(f, fft)
            
            spec = np.zeros((len(fft), self.n))
            spec[:, xmin_sample:xmin_sample + window_size] = np.asarray([fft] * window_size).transpose()
            extent = (0, (self.n / self.rate) * 1000, f[-1], f[0])
            im = self.ax3.imshow(spec, aspect='auto', extent=extent,
                    cmap = 'jet', norm = LogNorm(vmin=10**-12,vmax=10**-6))
            plt.gca().invert_yaxis()

            # make sure axis labels and titles do not overlap
            plt.tight_layout()
        
        # self.cb = self.fig.colorbar(im, ax=[self.ax3])
            
    def start(self):
        return self.box


def spectrogramAnimation(recorder=None):

    # the recorded sonud
    sound = recorder.sound
    rate = recorder.rate
    
    # spectrogram window length
    window_length = 256
    
    # numer of window
    n_windows = int(len(sound) / window_length)
    
    # compute the spectrogram
    f, t, Sxx = signal.spectrogram(sound, rate, nperseg=window_length, noverlap=0)
    
    # empty spectrogram 
    spec = np.zeros(Sxx.shape)
    
    # Y limit
    ylim = np.max(np.abs(sound)) 
    
    for i in range(n_windows):
    
        display.clear_output(wait=True)
        
        # initialize a figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # show the wave signal
        ax1.plot(np.linspace(0, len(sound), len(sound)) / rate, sound)
        ax1.set_xlim(0, len(sound)/rate)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')
        
        xmin = i*window_length / rate
        xmax = (i+1)*window_length / rate
        
        ax1.plot([xmin, xmin], [-ylim, ylim], 'orange')
        ax1.plot([xmax, xmax], [-ylim, ylim], 'orange')
        ax1.plot([xmin, xmax], [-ylim, -ylim], 'orange')
        ax1.plot([xmin, xmax], [ylim, ylim], 'orange')

        # show the spectrogram
        spec[:, : i + 1] = Sxx[:, : i + 1]
        extent = (t[0], t[-1], f[-1], f[0])
        im = ax2.imshow(spec, aspect='auto', extent=extent,
                cmap = 'jet', norm = LogNorm(vmin=10**-12,vmax=10**-6))
        plt.gca().invert_yaxis()
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [s]')
        
        plt.show()
        
        # time.sleep(0.1)