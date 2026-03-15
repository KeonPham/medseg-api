"""CLI tool to generate and register a new API key.

Usage:
    python scripts/create_api_key.py --name "test-client"
    python scripts/create_api_key.py --name "prod-service" --keys-file configs/api_keys.json
"""

import argparse
import hashlib
import json
import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEYS_FILE = Path("configs/api_keys.json")
KEY_LENGTH = 48  # 48 bytes → 64 chars of url-safe base64


def generate_api_key() -> str:
    """Generate a cryptographically secure API key.

    Returns:
        A url-safe random string.
    """
    return secrets.token_urlsafe(KEY_LENGTH)


def hash_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key.

    Args:
        api_key: The raw API key string.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def load_keys_file(path: Path) -> dict:
    """Load existing keys file or return empty structure.

    Args:
        path: Path to the JSON keys file.

    Returns:
        Dict with a 'keys' mapping.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"keys": {}}


def save_keys_file(path: Path, data: dict) -> None:
    """Write keys data to JSON file.

    Args:
        path: Path to the JSON keys file.
        data: The keys data structure to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    """Parse arguments, generate a key, and save its hash."""
    parser = argparse.ArgumentParser(description="Generate a new API key for MedSegAPI")
    parser.add_argument(
        "--name",
        required=True,
        help="Client name for this API key (e.g. 'test-client')",
    )
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=DEFAULT_KEYS_FILE,
        help=f"Path to the keys JSON file (default: {DEFAULT_KEYS_FILE})",
    )
    args = parser.parse_args()

    raw_key = generate_api_key()
    key_hash = hash_key(raw_key)

    data = load_keys_file(args.keys_file)
    data["keys"][key_hash] = {
        "name": args.name,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "active": True,
    }
    save_keys_file(args.keys_file, data)

    logger.info("")
    logger.info("API key created for '%s'", args.name)
    logger.info("Key: %s", raw_key)
    logger.info("")
    logger.info("Save this key now — it cannot be recovered.")
    logger.info("Hash stored in %s", args.keys_file)


if __name__ == "__main__":
    main()
