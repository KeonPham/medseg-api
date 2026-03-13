"""Download public chest X-ray datasets for lung segmentation training."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_montgomery(output_dir: Path) -> None:
    """Download the Montgomery County CXR dataset.

    Args:
        output_dir: Directory to save downloaded data.
    """
    raise NotImplementedError("Montgomery dataset download not yet implemented")


def download_shenzhen(output_dir: Path) -> None:
    """Download the Shenzhen Hospital CXR dataset.

    Args:
        output_dir: Directory to save downloaded data.
    """
    raise NotImplementedError("Shenzhen dataset download not yet implemented")


def download_jsrt(output_dir: Path) -> None:
    """Download the JSRT (Japanese Society of Radiological Technology) dataset.

    Args:
        output_dir: Directory to save downloaded data.
    """
    raise NotImplementedError("JSRT dataset download not yet implemented")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Data download script — datasets not yet configured")
