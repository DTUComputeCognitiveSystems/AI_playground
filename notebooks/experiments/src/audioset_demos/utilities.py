import numpy as np
import os
# import gzip
import h5py
import time
import logging
from scipy import stats
from sklearn import metrics
import re

try:
    import cPickle
except BaseException:
    import _pickle as cPickle


def digit(f):
    return int(re.compile(r"\d+").search(f).group(0))


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def create_logging(log_dir, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_path,
                        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime


def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf['x'][:]
        if hf['y'] is not None:
            y = hf['y'][:]
        else:
            y = hf['y']
        video_id_list = hf['video_id_list'][:].tolist()

    return x, y, video_id_list


def save_data(hdf5_path, x, y, video_id_list):
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('x', x, dtype='i8')
        hf.create_dataset('y', y, dtype='i8')
        hf.create_dataset('video_id_list', video_id_list, dtype='S11')

    return x, y, video_id_list

def find_model_to_load(path, restore_iteration=None):
    # Get model checkpoints in model directory
    models = [
        f.split('.')[0] for f in os.listdir(path) if f[-2:]=='h5'
    ]
    if not models:
        # Initialising graph
        start_iter = 0
        model_path = None
    else:
        model_iters = list(map(digit, models))
        if restore_iteration is not None:
            # Find the model closest to the restore_iteration
            iter_diffs = list(
                map(
                    (lambda x: abs(x - restore_iteration)),
                    model_iters
                )
            )
            min_iter_diff_ind = np.argmin(iter_diffs)
            start_iter = model_iters[min_iter_diff_ind]
            model_path = models[min_iter_diff_ind]
        else:
            # Restore latest model
            max_iter_ind = np.argmax(model_iters)
            start_iter = model_iters[max_iter_ind]
            model_path = models[max_iter_ind]
    model_path = os.path.join(path, model_path) + '.h5'
    return model_path, start_iter

def uint8_to_float32(x):
    return (np.float32(x) - 128.) / 128.


def bool_to_float32(y):
    return np.float32(y)


def transform_data(x, y=None):
    x = uint8_to_float32(x)
    if y is not None:
        y = bool_to_float32(y)
    return x, y


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats


def get_avg_stats(args):
    """Average predictions of different iterations and compute stats
    """

    test_hdf5_path = os.path.join(args.data_dir, "eval.h5")
    workspace = args.workspace
    filename = args.filename
    balance_type = args.balance_type
    model_type = args.model_type

    bgn_iteration = args.bgn_iteration
    fin_iteration = args.fin_iteration
    interval_iteration = args.interval_iteration

    get_avg_stats_time = time.time()

    # Load ground truth
    (test_x, test_y, test_id_list) = load_data(test_hdf5_path)
    target = test_y

    sub_dir = os.path.join(filename,
                           'balance_type={}'.format(balance_type),
                           'model_type={}'.format(model_type))

    # Average prediction probabilities of several iterations
    prob_dir = os.path.join(workspace, "probs", sub_dir, "test")
    prob_names = os.listdir(prob_dir)

    probs = []
    iterations = range(bgn_iteration, fin_iteration, interval_iteration)

    for iteration in iterations:

        pickle_path = os.path.join(prob_dir,
                                   "prob_{}_iters.p".format(iteration))

        prob = cPickle.load(open(pickle_path, 'rb'))
        probs.append(prob)

    avg_prob = np.mean(np.array(probs), axis=0)

    # Calculate stats
    stats = calculate_stats(avg_prob, target)

    logging.info("Callback time: {}".format(time.time() - get_avg_stats_time))

    # Write out to log
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])

    logging.info(
        "bgn_iteration, fin_iteration, interval_iteration: {}, {}, {}".format(
            bgn_iteration,
            fin_iteration,
            interval_iteration))

    logging.info("mAP: {:.6f}".format(mAP))
    logging.info("AUC: {:.6f}".format(mAUC))
    logging.info("d_prime: {:.6f}".format(d_prime(mAUC)))
