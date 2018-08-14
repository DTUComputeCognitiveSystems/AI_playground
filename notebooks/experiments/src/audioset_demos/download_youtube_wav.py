from __future__ import unicode_literals
import youtube_dl
import os
import numpy as np
import librosa
from scipy.io import wavfile
#import multiprocessing as mp
#import multiprocessing_logging
#import logging.handlers
#worker_args = [ytid, ts_start, ts_end, data_dir, ffmpeg_path, ffprobe_path]
#    pool.apply_async(partial(segment_mp_worker, **ffmpeg_cfg), worker_args)

MAXINT16 = np.iinfo(np.int16).max

extensionsToCheck = ['.m4a', '.webm', '.part', '.wav', '.ytdl']

#yt_embed_url_form = "http://www.youtube.com/embed/{video_id}?start={start_sec}&end={end_sec}"
yt_full_url_form = "http://www.youtube.com/watch?v={video_id}"
# yt_embed_url_form = "http://www.youtube.com/v/{video_id}?version=3&start={start_sec}&end={end_sec}&autoplay=0&hl=en_US&rel=0"

class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


def download_youtube_wav(video_id,
                        raw_dir=None,
                        short_raw_dir=None,
                        start_sec=0,
                        duration=10,
                        sample_rate=16000):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #if start_sec is not None and end_sec is not None:
        # ydl.download(
        #     [
        #         yt_embed_url_form.format(
        #             video_id=video_id,
        #             start_sec=int(start_sec),
        #             end_sec=int(end_sec)
        #         )
        #     ]
        # )
        # else:
        try: 
            ydl.download(
                [yt_full_url_form.format(video_id=video_id)]
            )
        except:
            print(video_id, 'not available for download anymore.')

    save_full_raw = raw_dir is not None and isinstance(raw_dir, str)
    save_short_raw = short_raw_dir is not None and isinstance(short_raw_dir, str)
    video_title = None

    for filename in os.listdir('.'):
        if video_id in filename:
            if '.wav' in filename:
                video_title = filename[:-16]
                if save_short_raw:
                    new_short_raw_name = os.path.join(
                        short_raw_dir,
                        filename[-15:]
                    )

                    print(
                        video_title
                        ,
                        '\n\tshort clip saved in:\n\t\t',
                        os.path.join(*new_short_raw_name.split('/')[-3:])
                    )
                
                    # Load and downsample audio to sample_rate
                    # audio is a 1D time series of the sound
                    # can also use (audio, fs) = soundfile.read(audio_path)
                    (audio, fs) = librosa.load(
                        filename,
                        sr=sample_rate,
                        offset=start_sec,
                        duration=duration
                    )
                    
                    # Store downsampled 10sec clip under data/short_raw/
                    wavfile.write(
                        filename=new_short_raw_name,
                        rate=sample_rate,
                        data=(audio * MAXINT16).astype(np.int16)
                    )

                if save_full_raw:
                    new_raw_name = os.path.join(
                        raw_dir,
                        filename[-15:]
                    )
                    print(
                        video_title,
                        '\n\tfull clip saved in:\n\t\t',
                        os.path.join(*new_raw_name.split('/')[-3:])
                    )
                    os.rename(filename, new_raw_name)

                else:
                    os.remove(filename)
            elif any(ext in filename for ext in extensionsToCheck):
                os.remove(filename)

    return video_title

#with wave.open(video_id + '.wav') as wf:

# youtube-dl -o "%(id)s.wav" -f 'bestaudio/best' --audio-format wav BaW_jenozKc