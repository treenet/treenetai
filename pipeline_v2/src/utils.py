"""Utility functions."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_file: Path = None,
    name: str = None,
    verbose: bool = True
) -> logging.Logger:
    """
    Setup logging configuration with a single log file.
    
    Creates a logger that writes to both console and file (if specified).
    The log file captures all output with timestamps.
    
    Args:
        log_file: Path to the log file. If None, only console output.
        name: Logger name (default: 'treenet')
        verbose: If True, set log level to INFO, else WARNING
        
    Returns:
        Logger instance ready to use
    """
    if name is None:
        name = 'treenet'
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler - simple format for readability
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - detailed format with timestamps
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        # Write header to file
        logger.debug(f"Log started at {datetime.now().isoformat()}")
        logger.debug(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str = 'treenet') -> logging.Logger:
    """
    Get existing logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
