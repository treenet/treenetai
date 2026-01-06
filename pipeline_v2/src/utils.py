"""Utility functions."""

import logging
from pathlib import Path


def setup_logging(verbose: bool = True, log_file: Path = None):
    """
    Setup logging configuration.
    
    Args:
        verbose: If True, set log level to INFO, else WARNING
        log_file: Optional log file path
    """
    level = logging.INFO if verbose else logging.WARNING
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


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
