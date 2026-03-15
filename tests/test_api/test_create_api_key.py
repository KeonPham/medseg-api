"""Tests for the create_api_key CLI script."""

import json
import subprocess
import sys

import pytest


@pytest.fixture()
def keys_file(tmp_path):
    return tmp_path / "api_keys.json"


def test_creates_key_file(keys_file) -> None:
    """Running the script should create the keys file with one entry."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_api_key.py",
            "--name",
            "test-client",
            "--keys-file",
            str(keys_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "API key created" in result.stderr or "API key created" in result.stdout

    data = json.loads(keys_file.read_text())
    assert len(data["keys"]) == 1
    entry = next(iter(data["keys"].values()))
    assert entry["name"] == "test-client"
    assert entry["active"] is True


def test_appends_to_existing(keys_file) -> None:
    """Running the script twice should add two separate keys."""
    for name in ["client-a", "client-b"]:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_api_key.py",
                "--name",
                name,
                "--keys-file",
                str(keys_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    data = json.loads(keys_file.read_text())
    assert len(data["keys"]) == 2
    names = {v["name"] for v in data["keys"].values()}
    assert names == {"client-a", "client-b"}


def test_prints_key_once(keys_file) -> None:
    """The raw key should appear in the script output."""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/create_api_key.py",
            "--name",
            "show-key",
            "--keys-file",
            str(keys_file),
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert "Key:" in output
    # Key line should contain a long random string
    for line in output.splitlines():
        if line.strip().startswith("Key:"):
            key_val = line.split("Key:", 1)[1].strip()
            assert len(key_val) > 30
