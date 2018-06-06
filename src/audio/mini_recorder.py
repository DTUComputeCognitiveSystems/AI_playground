import pyaudio
import numpy as np

class miniRecorder:
    def __init__(self, seconds=1, rate=44100):
        
        # self.format = pyaudio.paInt16
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = rate               # sampling rate (Hz)
        self.framesize = 2**10         # buffer size, number of data points to read at a time
        self.record_seconds = seconds  # how long should the recording be
        self.noframes = int((rate * seconds) / self.framesize)  # number of frames needed
        
        # instantiate pyaudio
        self.p = pyaudio.PyAudio()
        
        # instantiate stream
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.framesize)
        
    def record(self):
        
        # discard one frame before recording
        discard = self.stream.read(self.framesize)
        
        print("Recording...")

        data = self.stream.read(self.noframes * self.framesize)
        self.sound = np.frombuffer(data, dtype=np.float32)

        print("Finished recording...")
        
        # stop recording
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        return self.sound
    
    def playback(self):
        
        # instantiate pyaudio
        p = pyaudio.PyAudio()
        
        # instantiate stream
        stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  output=True,
                                  )
        
        try:
            stream.write(self.sound, len(self.sound))
        except:
            print("Error:", sys.exc_info()[0])
            
        stream.stop_stream()
        stream.close()
        p.terminate()