"""DICOM file handling utilities for medical image processing."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def read_dicom(file_path: str | Path) -> np.ndarray:
    """Read a DICOM file and return pixel data as numpy array.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        Pixel data as numpy array normalized to [0, 255] uint8.
    """
    import pydicom

    ds = pydicom.dcmread(str(file_path))
    pixel_array = ds.pixel_array.astype(np.float32)

    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
    pixel_array = (pixel_array * 255).astype(np.uint8)

    logger.info("Read DICOM: %s, shape=%s", file_path, pixel_array.shape)
    return pixel_array


def is_dicom(file_path: str | Path) -> bool:
    """Check if a file is a valid DICOM file.

    Args:
        file_path: Path to check.

    Returns:
        True if the file is a valid DICOM file.
    """
    path = Path(file_path)
    return path.suffix.lower() in (".dcm", ".dicom") or path.suffix == ""
