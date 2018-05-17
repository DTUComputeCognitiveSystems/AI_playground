import codecs
from pathlib import Path


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