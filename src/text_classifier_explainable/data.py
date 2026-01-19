"""Work with data: download, save, join text strings and clean text."""

import re
from pathlib import Path
import logging
from typing import Iterable, Optional
import urllib
import tarfile

logger = logging.getLogger(__name__)

LABELS = {
    'neg': 0, 
    'pos': 1
    }

data_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _validate_data_structure(search_path: Path) -> None:
    missing = [label for label in LABELS if not (search_path / label).is_dir()]
    if missing:
        raise FileNotFoundError(
            f'Missing label directories: {missing} in search_path'
        )


def join_data(search_path: Path, skip_errors: bool = False) -> tuple[list[str], list[int]]:
    """
    Load and join text data.

    Expected structure:
        search_path/
            pos/*.txt
            neg/*.txt
    
    Args:
        search_path: Path to dataset root directory.
        skip_errors: Skip unreadable files if True.
    
    Returns:
        texts: List of documents
        labels: List of corresponding numeric labels
    
    Raises:
        FileNotFoundError: If data structure is invalid.
        Exception: If there are troubles with reading text files.

    """

    _validate_data_structure(search_path)

    texts: list[str] = []
    labels: list[int] = []

    for label_name, label_id in LABELS.items():
        files = sorted((search_path / label_name).glob('*.txt'))
        for file in files:
            try:
                text = file.read_text(encoding='utf-8')
            except Exception as e:
                logger.warning('Failed to read file %s: %s', file, e)
                if not skip_errors:
                    raise
                continue
            texts.append(text)
            labels.append(label_id)

    logger.info('Loaded %d files', len(texts))

    return texts, labels


def save_data(path: Path, data: Iterable[str]) -> None:
    """
    Save an iterable of text strings into a single UTF-8 encoded file.

    Args:
        path: Full path including filename where the file will be saved.
        data: Iterable of strings to write to the file.

    Raises:
        OSError: If the file cannot be created or written to.
    """
    
    logger.info('Saving data to %s', path)

    count = 0
    try:
        with path.open('w', encoding='utf-8') as data_file:
            for text in data:
                data_file.write(text + '\n')
                count += 1
    except OSError:
        logger.exception('Failed to save to %s', path)
        raise

    logger.info('Saved %d lines to %s', count, path)


def clean_text(text: Optional[str]) -> str:
    """
    Clean a text string.

    Steps:
    1. Convert text to lowercase.
    2. Replace HTML tags with a space.
    3. Replace common punctuation characters (-().,:;?!) with a space.
    4. Remove all non-alphabetic characters.
    5. Normalize whitespace: multiple spaces -> single space.

    Args:
        text: A text string or None.
    
    Returns:
        A cleaned text string suitable for vectorization. If input text string is None or empty, returns empty string.
    """

    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[-().,:;?!]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def download_data(url: str, path: Path) -> None:
    """
    Download a file from a URL and save it to the given path.

    Args:
        url: URL of the file to download.
        path: Full path including filename where the file will be saved.

    Raises:
        urllib.error.URLError: If the download fails.
        OSError: If the file cannot be saved.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info('File already exists at %s, skipping download.', path)
        return

    try:
        logger.info('Downloading %s to %s', url, path)
        urllib.request.urlretrieve(url, path)
        logger.info('Download complete: %s', path)
    except urllib.error.URLError as e:
        logger.exception("Failed to download from %s: %s", url, e)
        raise
    except OSError as e:
        logger.exception("Failed to save file %s: %s", path, e)
        raise


def extract_archive(path_from: Path, path_to: Path):
    """
    Extract a tar.gz archive from path_from to path_to directory.

    Args:
        path_from: Path to the tar.gz archive.
        path_to: Directory where files will be extracted.

    Raises:
        tarfile.TarError: If the archive is corrupted or cannot be read.
        OSError: If there is a filesystem error during extraction.
    """

    path_to.mkdir(parents=True, exist_ok=True)

    if any(path_to.iterdir()):
        logger.info('Directory %s already has files, skipping extraction.', path_to)
        return
    
    try:
        logger.info('Extracting %s to %s', path_from, path_to)
        with tarfile.open(path_from, 'r:gz') as tar:
            tar.extractall(path=path_to)
        logger.info('Extraction complete: %s', path_to)
    except (tarfile.TarError, OSError) as e:
        logger.exception("Failed to extract %s", path_from)
        raise