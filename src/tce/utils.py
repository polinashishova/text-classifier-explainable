"""Setup logging and load json files."""

import logging
import json
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO, log_dir: Path = Path('logs'), log_filename: str = 'logs.log'):
    """
    Set up centralized logging for the project with console and file output.

    Args:
        level (int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir (Path): Directory to store log files.
        log_filename (str): Name of the log file.

    Returns:
        logging.Logger: Configured root logger.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('Logging started. Logs will be written to %s', logfile)

    return logger


def load_json(path: Path) -> dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        path (Path): Path to a JSON file.

    Returns:
        (dict) Parsed JSON content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If reading the file fails.
    """

    logger.info('Loading JSON from %s', path)

    if not path.exists():
        logger.error('JSON file not found: %s', path)
        raise FileNotFoundError(f'File not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            logger.info('Successfully loaded JSON from %s', path)
            return data

    except json.JSONDecodeError:
        logger.exception('Invalid JSON format in %s', path)
        raise

    except OSError:
        logger.exception('Failed to read JSON file %s', path)
        raise


def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data (dict[str, Any]): Data to be saved as JSON.
        path (Path): Path to save the JSON file.

    Raises:
        TypeError: If data is not JSON serializable.
        OSError: If writing the file fails.
    """

    logger.info('Saving JSON to %s', path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        logger.info('Successfully saved JSON to %s', path)

    except TypeError:
        logger.exception('Data is not JSON serializable: %s', path)
        raise

    except OSError:
        logger.exception('Failed to write JSON file %s', path)
        raise
