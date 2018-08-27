import codecs
import gzip
import pickle
import sys
from pathlib import Path

from tqdm import tqdm, trange, tqdm_notebook, tnrange

PICKLING_BYTES_STEP_SIZE = 100000


def get_dir(path):
    """
    Returns the directory of a file, or simply the original path if the path is a directory (has no extension)
    :param Path path:
    :return: Path
    """
    extension = path.suffix
    if extension == '':
        return path
    else:
        return path.parent


def ensure_directory(*arg):
    """
    Ensures the existence of a folder. If the folder does not exist it is created, otherwise nothing happens.
    :param str | Path arg: Any number of strings of Path-objects which can be combined to a path.
    """
    if len(arg) == 0:
        raise Exception("No input to ensure_folder")
    path = get_dir(Path(*arg))
    path.mkdir(parents=True, exist_ok=True)


def raw_buffered_line_counter(path, encoding="utf-8", buffer_size=1024 * 1024):
    """
    Fast way to count the number of lines in a file.
    :param Path path: Path to file.
    :param str encoding: Encoding used in file.
    :param int buffer_size: Size of buffer for loading.
    :return: int
    """
    # Open file
    f = codecs.open(str(path), encoding=encoding, mode="r")

    # Reader generator
    def _reader_generator(reader):
        b = reader(buffer_size)
        while b:
            yield b
            b = reader(buffer_size)

    # Reader used
    file_read = f.raw.read

    # Count lines
    line_count = sum(buf.count(b'\n') for buf in _reader_generator(file_read)) + 1

    return line_count


if not sys.stdout.isatty():
    trange = tnrange
    tqdm = tqdm_notebook


def save_as_compressed_pickle_file(obj, file_path, title=None):
    with gzip.open(file_path, "wb") as compressed_file:
        serialisation = pickle.dumps(obj)
        for start in trange(0, len(serialisation), PICKLING_BYTES_STEP_SIZE,
                            desc=title):
            end = start + PICKLING_BYTES_STEP_SIZE
            bytes_to_write = serialisation[start:end]
            compressed_file.write(bytes_to_write)


def load_from_compressed_pickle_file(file_path, title=None):

    with gzip.open(file_path, "rb") as compressed_file:
        obj = pickle.load(compressed_file)

    # TODO Speed up this method significantly before using it
    # compressed_size = file_path.stat().st_size
    #
    # with file_path.open("rb") as compressed_file:
    #     with gzip.GzipFile(fileobj=compressed_file) as uncompressed_file:
    #
    #         deserialisation = b""
    #         total_compressed_bytes_read_at_last_batch = 0
    #
    #         with tqdm(desc="", total=compressed_size, unit="B",
    #                   unit_scale=True) as progress_bar:
    #             for line_number, line in enumerate(uncompressed_file):
    #
    #                 deserialisation += line
    #
    #                 if event_number % 1000 == 0:
    #                     total_compressed_bytes_read = \
    #                         compressed_file.tell()
    #                     compressed_bytes_read_for_batch = \
    #                         total_compressed_bytes_read \
    #                         - total_compressed_bytes_read_at_last_batch
    #                     total_compressed_bytes_read_at_last_batch = \
    #                         total_compressed_bytes_read
    #                     progress_bar.update(
    #                         compressed_bytes_read_for_batch)
    #
    #         obj = pickle.loads(deserialisation)

    return obj
