
import numpy as np
import os
import sys
import pandas as pd

from tqdm import tqdm

# Multiprocessing and threading
import multiprocessing as mp

# Our YouTube video downloader based on youtube-dl module
import download_youtube_wav as dl_yt
import utilities

# from AudioSetClassifier import AudioSetClassifier

SAMPLE_RATE = 16000
# predictor = AudioSetClassifier()
pbar = tqdm(total=22200)

def download_and_embed(
    video_id,
    raw_dir,
    short_raw_dir,
    start_sec,
    duration,
    sample_rate,
    video_ids_incl,
    video_id_bin,
    video_ids_incl_bin,
    embeddings,
    label_inds,
    labels):
    #progress_bar):
    
    if video_id + '.wav' not in os.listdir(short_raw_dir):
        dl_yt.download_youtube_wav(
            video_id=video_id,
            raw_dir=raw_dir,
            short_raw_dir=short_raw_dir,
            start_sec=start_sec,
            duration=duration,
            sample_rate=sample_rate
        )

    if video_id + '.wav' in os.listdir(short_raw_dir):
        wav_file = os.path.join(short_raw_dir, video_id) + '.wav'
        video_ids_incl.append(video_id)
        video_ids_incl_bin.append(video_id_bin)
        embeddings.append(predictor.embed(wav_file))
        class_vec = np.zeros([1, 527])
        class_vec[:, label_inds] = 1
        labels.append(class_vec)

        #os.remove(wav_file)

    pbar.update(1)


def download_wav(
    video_id,
    raw_dir,
    short_raw_dir,
    start_sec,
    duration,
    sample_rate):

    if video_id + '.wav' not in os.listdir(short_raw_dir):
        dl_yt.download_youtube_wav(
            video_id=video_id,
            raw_dir=raw_dir,
            short_raw_dir=short_raw_dir,
            start_sec=start_sec,
            duration=duration,
            sample_rate=sample_rate
        )
    pbar.update(1)


def retrieve_embeddings(
    data_path=None,
    metadata_file=None,
    class_labels_file=None,
    embed_path=None):

    if metadata_file is None:
        metadata_file = os.path.join(
            data_path,
            'metadata',
            'balanced_train_segments.csv'
        )
    if class_labels_file is None:
        class_labels_file = os.path.join(
            data_path,
            'metadata',
            'class_labels_indices.csv'
        )
    if embed_path is None:
        embed_path = os.path.join(
            data_path,
            'data',
            'embeddings',
        )

    if 'eval' in metadata_file:
        embed_file = os.path.join(embed_path, 'eval.h5')
    elif 'unbalanced' in metadata_file:
        embed_file = os.path.join(embed_path, 'unbal_train.h5')
    elif 'balanced' in metadata_file:
        embed_file = os.path.join(embed_path, 'bal_train.h5')

    class_labels = pd.read_csv(
        os.path.join(
            data_path,
            'metadata',
            'class_labels_indices.csv'
        )
    )

    colnames = '# YTID, start_seconds, end_seconds, positive_labels'.split(', ')
    metadata = pd.read_csv(metadata_file, sep=', ', header=2) #usecols=colnames)
    metadata.rename(columns={colnames[0]: colnames[0][-4:]}, inplace=True)
    metadata['pos_lab_list'] = metadata.positive_labels.apply(lambda x: x[1:-1].split(','))
    colnames.extend('pos_lab_list')


    # Prepare download directories
    short_raw_dir = os.path.join(data_path, 'data', 'short_raw')
    if not os.path.exists(short_raw_dir):
        os.makedirs(short_raw_dir)
    raw_dir = os.path.join(data_path, 'data', 'raw')
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(embed_path):
        os.makedirs(embed_path)


    video_ids = metadata.YTID.tolist()
    video_ids_bin = metadata.YTID.astype('|S11').tolist()
    video_start_time = metadata.start_seconds.tolist()
    video_end_time = metadata.end_seconds.tolist()

    class_dict = class_labels.set_index('mid').T.to_dict('list')

    metadata['pos_lab_ind_list'] = metadata.pos_lab_list.apply(
        lambda x: [class_dict[y][0] for y in x]
    )

    video_ids_incl_bin = []
    video_ids_incl = []
    labels = []
    embeddings = []
    # i = 0
    # jobs = []
    # while video_ids:
    #     if sum([j.is_alive() for j in jobs]) < 100:
    #         #print(i, video_ids[i])
    #         # Download and store audio and embeddings from video_id
    #         args = (
    #             video_ids[i],
    #             None,
    #             short_raw_dir,
    #             video_start_time[i],
    #             video_end_time[i] - video_start_time[i],
    #             SAMPLE_RATE,
    #             predictor,
    #             video_ids_incl,
    #             video_ids_bin[i],
    #             video_ids_incl_bin,
    #             embeddings,
    #             metadata.loc[i, 'pos_lab_ind_list'],
    #             labels,
    #             pbar
    #         )
    #         process = multiprocessing.Process(
    #             target=download_and_embed,
    #             args=args
    #         )
    #         jobs.append(process)
    #         i += 1
    #         process.start()
            
            
    #     else:
    #         # Number of present jobs
    #         n_jobs = len(jobs)
    #         jobs = [j for j in jobs if j.is_alive()]
    #         # Update progress bar with number of terminated jobs 
    #         # not in the list anymore
    #         pbar.update(n_jobs - len(jobs))
    #         # Ensure all of the processes have finished
    #         for j in jobs:
    #             j.join()


    download_process_args = []
    for i, video_id in enumerate(video_ids):
        download_process_args.append(
            (
                video_id,
                None,
                short_raw_dir,
                video_start_time[i],
                video_end_time[i] - video_start_time[i],
                SAMPLE_RATE
                # video_ids_incl,
                # video_ids_bin[i],
                # video_ids_incl_bin,
                # embeddings,
                # metadata.loc[i, 'pos_lab_ind_list'],
                # labels
                # pbar
            )
        )

    n_proc = 16
    n_cpus = mp.cpu_count()
    print(
        "Starting pool with {} workers out of {} available CPU's".format(
            n_proc,
            n_cpus
        )
    )
    pool = mp.Pool(n_proc, maxtasksperchild=32)
    pool.starmap(download_wav, download_process_args)
    pool.close()
    pool.join()

    pbar.close()


    # for i, video_id in enumerate(video_ids):
    #     # Download and store audio and embeddings from video_id
    #     if video_id + '.wav' not in os.listdir(short_raw_dir):
    #         args = (
    #             video_id,
    #             None,
    #             short_raw_dir,
    #             video_start_time[i],
    #             video_end_time[i]-video_start_time[i],
    #             SAMPLE_RATE,
    #             predictor,
    #             video_ids_incl,
    #             video_ids_bin[i],
    #             video_ids_incl_bin,
    #             embeddings,
    #             metadata.loc[i, 'pos_lab_ind_list'],
    #             labels
    #         )
    #         process = multiprocessing.Process(
    #             target=download_and_embed,
    #             args=args
    #         )
    #         jobs.append(process)


    # # Start the processes (i.e. calculate the random number lists)
    # for j in jobs:
    #     j.start()

    # # Ensure all of the processes have finished
    # for j in jobs:
    #     j.join()

    utilities.save_data(
        hdf5_path=embed_file,
        x=np.array(embeddings),
        video_id_list=np.array(video_ids_incl_bin),
        y=np.array(labels)
    )

def main():
    retrieve_embeddings(
        data_path=os.path.join('/nobackup', 'maxvo', 'audioset')
    )

if __name__ == "__main__":
    main()
    
