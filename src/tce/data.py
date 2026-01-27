"""Work with data: download, save, join text string, load and save data and clean text."""

import re
import logging
import urllib.request
import urllib.error
import tarfile
from typing import Iterable, Optional
from scipy import sparse
from pathlib import Path

logger = logging.getLogger(__name__)

LABELS: dict[str, int] = {
    'neg': 0, 
    'pos': 1
    }


def download_data(url: str, path: Path) -> None:
    """
    Download a file from a URL and save it to the given path.
    If file alredy exists at the given path, download is skipped.

    Args:
        url (str): URL of the file to download.
        path (Path): Full path including filename where the file will be saved.

    Raises:
        urllib.error.URLError: If the download fails.
        OSError: If the file cannot be saved.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info('File already exists at %s, skipping download', path)
        return

    try:
        logger.info('Downloading %s to %s', url, path)
        urllib.request.urlretrieve(url, path)
        logger.info('Download complete: %s', path)
    except urllib.error.URLError:
        logger.exception('Failed to download from %s: %s', url)
        raise
    except OSError:
        logger.exception('Failed to save file %s: %s', path)
        raise


def extract_archive(path_from: Path, directory: Path, expected_path: Path) -> None:
    """
    Extract a tar.gz archive from path_from to directory unless expected extracted data already exists.

    Args:
        path_from (Path): Path to the tar.gz archive.
        directory (Path): Directory where files will be extracted.
        expected_path (Path): Path that should exist after successful extraction.

    Raises:
        tarfile.TarError: If the archive is corrupted or cannot be read.
        OSError: If there is a filesystem error during extraction.
    """

    directory.mkdir(parents=True, exist_ok=True)

    if expected_path.exists():
        logger.info('Archive already extracted, found %s. Skipping extraction', expected_path)
        return

    try:
        logger.info('Extracting %s to %s', path_from, directory)
        with tarfile.open(path_from, 'r:gz') as tar:
            tar.extractall(path=directory)
        logger.info('Extraction complete: %s', directory)
    except (tarfile.TarError, OSError):
        logger.exception('Failed to extract %s', path_from)
        raise


def _validate_data_structure(search_path: Path) -> None:
    missing = [label for label in LABELS if not (search_path / label).is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing label directories: {missing} in search_path')


def join_data(search_path: Path, skip_errors: bool = False) -> tuple[list[str], list[int]]:
    """
    Load and join text data.

    Expected structure:
        search_path/
            pos/*.txt
            neg/*.txt
    
    Args:
        search_path (Path): Path to dataset root directory.
        skip_errors (bool): Skip unreadable files if True.
    
    Returns:
        texts (list[str]): List of documents.
        labels (list[int]): List of corresponding numeric labels.
                Always integers: 0 for 'neg', 1 for 'pos'.
    
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
            except Exception:
                logger.warning('Failed to read file %s: %s', file)
                if not skip_errors:
                    raise
                continue
            texts.append(text)
            labels.append(label_id)

    logger.info('Loaded %d files', len(texts))

    return texts, labels


def save_data(path: Path, data: Iterable[str | int] | sparse.spmatrix) -> None:
    """
    Save text data, labels or a SciPy sparse matrix to disk.
    If file exists, saving is skipped.

    - Text data or data labels is saved as UTF-8 lines.
    - Sparse matrices are saved in .npz format.

    Args:
        path (Path): Full path including filename.
        data (Iterable[str | int]): Iterable of strings or integers or SciPy sparse matrix.

    Raises:
        OSError: If writing fails.
    """
    
    logger.info('Saving data to %s', path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info('File %s is already exists', path)
        return

    if sparse.isspmatrix(data):

        if path.suffix != '.npz':
            path = path.with_suffix('.npz')

        try:
            sparse.save_npz(path, data)
            logger.info('Saved sparse matrix %s with shape %s and nnz=%d', data.__class__.__name__, data.shape, data.nnz)
            return
        except OSError:
            logger.exception('Failed to save to %s', path)
            raise

    if isinstance(data, str):
        raise TypeError('Expected iterable of strings, got single string')
    elif not data:
        raise ValueError(f'No data to save to {path}')
    
    count = 0
    try:
        with path.open('w', encoding='utf-8') as data_file:
            for text in data:
                data_file.write(str(text) + '\n')
                count += 1
    except OSError:
        logger.exception('Failed to save to %s', path)
        raise

    logger.info('Saved %d lines to %s', count, path)


def load_data(path: Path) -> list[str | int] | sparse.spmatrix:
    """
    Load text data, data labels or a SciPy sparse matrix from disk.

    - Text data is loaded as a list of UTF-8 strings (one per line).
    - Data labels are loaded as list of integers.
    - Sparse matrices are loaded from .npz format.

    Args:
        path (Path): Full path including filename.

    Returns:
        list[str] - texts, list[int] - labels or scipy.sparse.spmatrix:
            - List of text lines if a text file is provided.
            - List (int) of labels if a data labels file is povided.
            - SciPy sparse matrix if a .npz file is provided.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If reading fails.
    """

    logger.info('Loading data from %s', path)

    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    if path.suffix == '.npz':
        try:
            data = sparse.load_npz(path)
            logger.info(
                'Loaded sparse matrix %s with shape %s and nnz=%d',
                data.__class__.__name__,
                data.shape,
                data.nnz
            )
            return data
        except OSError:
            logger.exception('Failed to load sparse matrix from %s', path)
            raise

    try:
        with path.open('r', encoding='utf-8') as data_file:
            lines = [line.rstrip('\n') for line in data_file]
        if all(line.isdigit() for line in lines):
            lines = [int(line) for line in lines]
        logger.info('Loaded %d lines from %s', len(lines), path)
        return lines
    except OSError:
        logger.exception('Failed to load text data from %s', path)
        raise


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
        text (str): A text string or None.
    
    Returns:
        A cleaned text string suitable for vectorization. 
        If input text string is None or empty, returns empty string.
    """

    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[-().,:;?!]', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def process_split(raw_texts_dir: Path, raw_texts_path: Path, labels_path: Path, proc_texts_path: Path) -> None:
    """
    Process a dataset split: join raw texts, clean them and save texts and labels.
    Skips processing if output files already exist.

    Args:
        raw_texts_dir (Path): Directory with texts files (.txt).
        raw_texts_path (Path): Path for saving texts to a single .txt file.
        labels_path (Path): Path for saving corresponding labels.
        proc_texts_path (Path): Path for saving cleaned texts.
    """

    if raw_texts_path.exists() and labels_path.exists() and proc_texts_path.exists():
        logger.info('Split %s already processed. Skipping', raw_texts_dir.name)
        return
    
    logger.info('Processing split %s', raw_texts_dir.name)

    raw_texts, labels = join_data(raw_texts_dir)
    save_data(raw_texts_path, raw_texts)
    save_data(labels_path, labels)
    texts_cleaned = [clean_text(text) for text in raw_texts]
    save_data(proc_texts_path, texts_cleaned)