"""API key authentication via FastAPI dependency."""

import hashlib
import json
import logging
from pathlib import Path

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

DEFAULT_KEYS_PATH = Path("configs/api_keys.json")


def _hash_key(api_key: str) -> str:
    """Return the SHA-256 hex digest of an API key."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def _load_keys(path: Path = DEFAULT_KEYS_PATH) -> dict[str, dict]:
    """Load the API key registry from a JSON file."""
    if not path.exists():
        logger.warning("API keys file not found: %s", path)
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("keys", {})


def _save_keys(path: Path, keys: dict[str, dict]) -> None:
    """Persist the key registry back to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"keys": keys}, f, indent=2)
        f.write("\n")


class APIKeyValidator:
    """Callable FastAPI dependency that validates X-API-Key headers.

    Supports two key types:
    - **master**: unlimited usage (no ``max_uses`` field or ``max_uses: null``)
    - **guest**: limited to ``max_uses`` predictions, tracked via ``used`` counter
    """

    def __init__(
        self,
        keys_path: Path = DEFAULT_KEYS_PATH,
        enabled: bool = True,
    ) -> None:
        self._keys_path = keys_path
        self._enabled = enabled
        self._keys: dict[str, dict] = {}
        self._reload_keys()

    def _reload_keys(self) -> None:
        """Reload keys from disk."""
        self._keys = _load_keys(self._keys_path)
        logger.info("Loaded %d API key(s) from %s", len(self._keys), self._keys_path)

    def reload(self) -> None:
        """Public method to force a key reload (e.g. after adding a new key)."""
        self._reload_keys()

    def _persist(self) -> None:
        """Write current key state to disk (for usage counter updates)."""
        _save_keys(self._keys_path, self._keys)

    def validate(self, api_key: str) -> str:
        """Validate an API key string.

        Returns:
            The client name associated with the key.

        Raises:
            HTTPException: 401 if the key is invalid or disabled.
            HTTPException: 403 if a guest key has exhausted its usage limit.
        """
        if not self._enabled:
            return "anonymous"

        key_hash = _hash_key(api_key)
        key_entry = self._keys.get(key_hash)

        if key_entry is None:
            logger.warning("Rejected invalid API key (hash=%s...)", key_hash[:12])
            raise HTTPException(status_code=401, detail="Invalid API key")

        if not key_entry.get("active", True):
            logger.warning("Rejected disabled API key: %s", key_entry.get("name"))
            raise HTTPException(status_code=401, detail="API key is disabled")

        # Check usage limit for guest keys
        max_uses = key_entry.get("max_uses")
        if max_uses is not None:
            used = key_entry.get("used", 0)
            if used >= max_uses:
                logger.warning(
                    "Guest key '%s' exhausted (%d/%d)",
                    key_entry.get("name"), used, max_uses,
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"API key usage limit reached ({used}/{max_uses}). Contact the admin for a new key.",
                )
            # Increment counter and persist
            key_entry["used"] = used + 1
            self._persist()
            logger.info(
                "Guest key '%s' used %d/%d",
                key_entry.get("name"), used + 1, max_uses,
            )

        return key_entry.get("name", "unknown")


# Module-level validator instance. Replaced at app startup and in tests.
_validator = APIKeyValidator(enabled=False)


def set_validator(validator: APIKeyValidator) -> None:
    """Replace the active validator (called at app startup and in tests).

    Args:
        validator: The new APIKeyValidator to use.
    """
    global _validator  # noqa: PLW0603
    _validator = validator


def verify_api_key(x_api_key: str = Header(...)) -> str:
    """FastAPI dependency that delegates to the current module-level validator.

    Args:
        x_api_key: The API key from the X-API-Key header.

    Returns:
        The client name associated with the key.
    """
    return _validator.validate(x_api_key)
