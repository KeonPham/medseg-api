"""CLI tool to generate and register a new API key.

Usage:
    # Master key (unlimited)
    python scripts/create_api_key.py --name "production"

    # Guest key (limited uses)
    python scripts/create_api_key.py --name "friend-john" --max-uses 5

    # One-time key
    python scripts/create_api_key.py --name "demo-user" --max-uses 1

    # List all keys
    python scripts/create_api_key.py --list

    # Revoke a key by name
    python scripts/create_api_key.py --revoke "friend-john"
"""

import argparse
import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_KEYS_FILE = Path("configs/api_keys.json")
KEY_LENGTH = 48


def generate_api_key() -> str:
    return secrets.token_urlsafe(KEY_LENGTH)


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def load_keys_file(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"keys": {}}


def save_keys_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def list_keys(path: Path) -> None:
    data = load_keys_file(path)
    keys = data.get("keys", {})
    if not keys:
        logger.info("No API keys found.")
        return
    logger.info("%-20s %-8s %-10s %-10s %s", "NAME", "ACTIVE", "TYPE", "USAGE", "CREATED")
    logger.info("-" * 75)
    for _hash, entry in keys.items():
        name = entry.get("name", "?")
        active = "yes" if entry.get("active", True) else "no"
        max_uses = entry.get("max_uses")
        used = entry.get("used", 0)
        if max_uses is not None:
            key_type = "guest"
            usage = f"{used}/{max_uses}"
        else:
            key_type = "master"
            usage = "unlimited"
        created = entry.get("created_at", "?")[:10]
        logger.info("%-20s %-8s %-10s %-10s %s", name, active, key_type, usage, created)


def revoke_key(path: Path, name: str) -> None:
    data = load_keys_file(path)
    keys = data.get("keys", {})
    found = False
    for _hash, entry in keys.items():
        if entry.get("name") == name:
            entry["active"] = False
            found = True
            break
    if found:
        save_keys_file(path, data)
        logger.info("Key '%s' has been revoked.", name)
    else:
        logger.info("No key found with name '%s'.", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage API keys for MedSegAPI")
    parser.add_argument("--name", help="Client name for this API key")
    parser.add_argument(
        "--max-uses", type=int, default=None,
        help="Max number of predictions allowed (omit for unlimited master key)",
    )
    parser.add_argument(
        "--keys-file", type=Path, default=DEFAULT_KEYS_FILE,
        help=f"Path to the keys JSON file (default: {DEFAULT_KEYS_FILE})",
    )
    parser.add_argument("--list", action="store_true", help="List all API keys")
    parser.add_argument("--revoke", metavar="NAME", help="Revoke a key by name")
    args = parser.parse_args()

    if args.list:
        list_keys(args.keys_file)
        return

    if args.revoke:
        revoke_key(args.keys_file, args.revoke)
        return

    if not args.name:
        parser.error("--name is required when creating a key")

    raw_key = generate_api_key()
    key_hash = hash_key(raw_key)

    data = load_keys_file(args.keys_file)
    entry = {
        "name": args.name,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "active": True,
    }
    if args.max_uses is not None:
        entry["max_uses"] = args.max_uses
        entry["used"] = 0

    data["keys"][key_hash] = entry
    save_keys_file(args.keys_file, data)

    key_type = f"guest ({args.max_uses} uses)" if args.max_uses else "master (unlimited)"
    logger.info("")
    logger.info("API key created for '%s' [%s]", args.name, key_type)
    logger.info("Key: %s", raw_key)
    logger.info("")
    logger.info("Save this key now — it cannot be recovered.")


if __name__ == "__main__":
    main()
