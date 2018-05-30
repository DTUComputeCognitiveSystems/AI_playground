from functools import lru_cache
from tempfile import NamedTemporaryFile
from zipfile import ZipFile
import fastText
from pathlib import Path

from src.utility.connectivity import download_file
from src.utility.files import ensure_directory

_data_dir = Path("data", "fasttext")
ensure_directory(_data_dir)


@lru_cache(maxsize=3)
def get_fasttext_model(lang="en"):
    bin_path = Path(_data_dir, "wiki.{}.bin".format(lang))
    zip_path = Path(_data_dir, "wiki.{}.zip".format(lang))
    print("Getting FastText data.")

    # Check for bin file
    if bin_path.exists():
        print("\tLoading model from binary file.")
        model = fastText.load_model(str(bin_path))

    # Check for zip file
    elif zip_path.exists():
        with ZipFile(str(zip_path)) as zip_file:
            print("\tUsing zip file.")

            # Create temporary file for unzipping
            temp_file = NamedTemporaryFile(delete=False)

            # For ensuring deletion of temporary file
            try:
                # Write
                print("\tUnzipping into temporary file: {}".format(temp_file.name))
                temp_file.write(zip_file.read(name="wiki.{}.bin".format(lang)))

                # Pass temporary file to fasttext
                print("\tLoading model from temporary file.")
                model = fastText.load_model(temp_file.name)

            # Delete temporary file
            finally:
                print("\tRemoving temporary file.")
                temp_file.close()
                temp_path = Path(temp_file.name)
                if temp_path.exists():
                    temp_path.unlink()

    else:
        # Consider downloading
        url = "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.zip".format(lang)
        downloaded = download_file(
            url=url,
            destination=_data_dir,
            prompt="Data for FastText not found. \nIt can be downloaded from {}. ".format(url) +
                   "\nDo you want to download this file now? - Note that this is a LARGE file."
                   "\n[y/N]",
            default_yes=False,
            chunk_size= 4096 * 32
        )

        # Check if downloaded
        if downloaded:
            model = get_fasttext_model(lang=lang)

        else:
            raise FileNotFoundError("Can not find FastText file for language {} in directory {}.".format(
                lang,
                _data_dir.resolve())
            )

    print("festText model loaded.")

    return model
