import sys
import urllib
from io import TextIOWrapper
from pathlib import Path

import requests
from tqdm import tqdm  # , tqdm_notebook
from tqdm._tqdm_notebook import tqdm_notebook

from src.utility.files import ensure_directory


def download_file(url, destination, prompt=None, require_accept=True, default_yes=True, chunk_size=4096*8):
    if require_accept:
        prompt = prompt if prompt is not None else "Do you wish to download file at {}? [Y/n]".format(url)
        answer = input(prompt)
        if default_yes:
            download = "n" not in answer.lower()
        else:
            download = "y" in answer.lower()

    else:
        print(prompt)
        download = True

    # Don't download
    if not download:
        return False

    # Get response from online and determine length or data
    response = requests.get(url=url, stream=True)
    total_length = response.headers.get('content-length')

    # Initialize and check for data
    data = [b""]
    if total_length is None:  # no content length header
        pass

    # Content exists
    else:
        # Integer length
        total_length = int(total_length)

        # Use tqdm for monitoring download progress
        print("Downloading:", flush=True)
        with tqdm(total=total_length) as p_bar:

            # Download each chunk
            for data_chunk in response.iter_content(chunk_size=chunk_size):
                p_bar.update(len(data_chunk))
                data.append(data_chunk)

        # Save it all
        sys.stderr.flush()
        sys.stdout.flush()
        print("Saving to file", flush=True)
        if isinstance(destination, (Path, str)):
            destination = Path(destination)

            # Check for complete file path
            if destination.suffix:
                with destination.open("wb") as file:
                    file.write(b"".join(data))

            # Otherwise assume directory
            else:
                ensure_directory(destination)

                # Get URL filename
                filename = Path(url).name

                # Store
                with Path(destination, filename).open("wb") as file:
                    file.write(b"".join(data))

        elif isinstance(destination, TextIOWrapper):
            destination.write(b"".join(data))

        else:
            raise ValueError("Unknown destination type: {}".format(type(destination).__name__))

    print("File downloaded.")
    return True


if not sys.stdout.isatty():
    tqdm = tqdm_notebook


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def retrieve_file(url, path, title=None):

    path = Path(path)
    ensure_directory(path.parent)

    site = urllib.request.urlopen(url)
    meta = site.info()
    total_size = int(meta["Content-Length"])

    with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                  desc=title, total=total_size) as t:
        urllib.request.urlretrieve(url, filename=path,
                                   reporthook=t.update_to, data=None)
