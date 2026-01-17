import os
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

LABELS = {
    'neg': 0, 
    'pos': 1
    }


def _validate_data_structure(search_path: Path) -> None:
    missing = [label for label in LABELS if not (search_path / label).is_dir()]
    if missing:
        raise FileNotFoundError(
            f'Missing label directories: {missing} in search_path'
        )


def join_data(search_path: Path, skip_errors: bool = False) -> tuple[list[str], list[int]]:
    '''
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

    '''

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

    logger.info(f'Loaded %d files', len(texts))

    return texts, labels


def save_data(file_name, path, data):
    with open(os.path.join(path, file_name), 'w', encoding='utf-8') as data_file:
        for text in data:
            data_file.write(text + '\n')


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[-().,:;?!]", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text