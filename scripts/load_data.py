"""
load_data.py

Script to download, extract, join, clean, and save raw dataset splits (train/test) 
for the IMDb dataset. Produces processed text files and labels ready for model training.

Key steps:
- Load paths configuration
- Download and extract dataset archive
- Join texts and labels for train/test splits
- Clean texts and save raw and processed versions

Usage:
    python scripts/load_data.py
"""

from tce.data import download_data, extract_archive, process_split
from tce.utils import setup_logging, load_json
from pathlib import Path

logger = setup_logging()


def main() -> None:
    
    paths_cfg_path = Path('config/paths.json')
    paths = load_json(paths_cfg_path)

    raw_dir = Path(paths['data_raw_dir'])
    proc_dir = Path(paths['data_processed_dir'])

    data_url = paths['data_url']
    arch_path = raw_dir / paths['aclImdb_v1']

    download_data(url=data_url, path=arch_path)

    ext_path = raw_dir / paths['aclImdb']

    extract_archive(arch_path, raw_dir, ext_path)

    raw_train_dir = ext_path / paths['train']
    raw_test_dir = ext_path / paths['test']

    train_texts_path = raw_dir / paths['raw_train']
    test_texts_path = raw_dir / paths['raw_test']

    train_labels_path = proc_dir / paths['train_labels']
    test_labels_path = proc_dir / paths['test_labels']

    train_cleaned_path = proc_dir / paths['processed_train']
    test_cleaned_path = proc_dir / paths['processed_test']

    process_split(raw_train_dir, train_texts_path, train_labels_path, train_cleaned_path)
    process_split(raw_test_dir, test_texts_path, test_labels_path, test_cleaned_path)


if __name__ == '__main__':
    main()