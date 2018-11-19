import pyaudio
import numpy as np
import wave

class miniRecorder:
    def __init__(self, seconds=1, rate=22050):# rate=44100
        
        # self.format = pyaudio.paInt16
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = rate               # sampling rate (Hz)
        self.framesize = 2205         # buffer size, number of data points to read at a time 2 ** 10
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

        self.data = self.stream.read(self.noframes * self.framesize)
        self.sound = np.frombuffer(self.data, dtype=np.float32)

        print("Finished recording...")
        
        # stop recording
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        
        return self.sound

    def write2file(self, fname):
        
        wavefile = wave.open(fname, 'wb')
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        wavefile.writeframes(self.data)

        wavefile.close()
    
    def playback(self, sound=None):

        if not sound:
            sound = self.sound

        # instantiate pyaudio
        p = pyaudio.PyAudio()
        
        # instantiate stream
        stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  output=True,
                                  )
        
        try:
            stream.write(sound, len(sound))
        except:
            print("Error:", sys.exc_info()[0])
            
        stream.stop_stream()
        stream.close()
        p.terminate()