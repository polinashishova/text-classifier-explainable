from tce.data import download_data, extract_archive, join_data, save_data, clean_text
from tce.utils import setup_logging, load_json
from pathlib import Path
import logging

logger = setup_logging()

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

raw_dir = Path(paths['data_raw_dir'])
proc_dir = Path(paths['data_processed_dir'])

data_url = paths['data_url']
arch_path = Path(raw_dir / paths['aclImdb_v1'])

download_data(url=data_url, path=arch_path)

ext_path = Path(raw_dir / paths['aclImdb'])

extract_archive(arch_path, raw_dir, ext_path)

raw_train_dir = Path(ext_path / paths['train'])
raw_test_dir = Path(ext_path / paths['test'])

train_texts_path = Path(raw_dir / paths['raw_train'])
test_texts_path = Path(raw_dir / paths['raw_test'])

train_labels_path = Path(proc_dir / paths['train_labels'])
test_labels_path = Path(proc_dir / paths['test_labels'])

train_cleaned_path = Path(proc_dir / paths['processed_train'])
test_cleaned_path = Path(proc_dir / paths['processed_test'])

if not all((train_texts_path.exists(), train_labels_path.exists(), train_cleaned_path.exists())):
    train_texts, train_labels = join_data(raw_train_dir)
    save_data(train_texts_path, train_texts)
    save_data(train_labels_path, train_labels)
    train_cleaned = [clean_text(text) for text in train_texts]
    save_data(train_cleaned_path, train_cleaned)

if not all((test_texts_path.exists(), test_labels_path.exists(), test_cleaned_path.exists())):
    test_texts, test_labels = join_data(raw_test_dir)
    save_data(test_texts_path, test_texts)
    save_data(test_labels_path, test_labels)
    test_cleaned = [clean_text(text) for text in test_texts]
    save_data(test_cleaned_path, test_cleaned)