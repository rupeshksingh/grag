# config/logging_config.py
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;32m',     # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m', # Purple
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Save original format
        format_orig = self._style._fmt

        # Add colors if running in terminal
        if sys.stdout.isatty():
            self._style._fmt = f"{self.COLORS.get(record.levelname, self.COLORS['RESET'])}" \
                             f"{format_orig}{self.COLORS['RESET']}"

        # Call original formatter
        result = super().format(record)

        # Restore original format
        self._style._fmt = format_orig
        return result

def get_log_file_path() -> Path:
    """Generate log file path with timestamp"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"tender_kg_{timestamp}.log"

def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    """Configure and return logger with both file and console handlers
    
    Args:
        log_file: Optional path to log file. If None, generates timestamped file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('TenderKG')
    logger.setLevel(getattr(logging, log_level))
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = CustomFormatter(
        '%(asctime)s - %(levelname)-8s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler
    log_file = log_file or get_log_file_path()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log initial setup information
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return logger

# Create default logger instance
logger = setup_logging()