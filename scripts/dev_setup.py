#!/usr/bin/env python3
"""Cross-platform local setup for Smritikosh.

Copies the example env files when they are missing, replaces a still-default
JWT_SECRET / AUTH_SECRET, and can set or clear BOOTSTRAP_ADMIN. Works in
cmd.exe and PowerShell as well as bash — no sed or openssl.

Usage (from the repository root):

    python scripts/dev_setup.py
    python scripts/dev_setup.py --bootstrap-admin
    python scripts/dev_setup.py --clear-bootstrap-admin
"""

from __future__ import annotations

import argparse
import re
import secrets
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

DEFAULT_SECRET = "change-me-in-production"
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_BOOTSTRAP_ASSIGNMENT = re.compile(r"^#?\s*BOOTSTRAP_ADMIN\s*=")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _split_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def fill_default_secret(text: str, key: str, secret: str) -> str:
    """Replace ``key=<default>`` with ``key=secret``. Leave any other value."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        body, ending = _split_ending(line)
        match = _ASSIGNMENT.match(body.strip())
        if match and match.group(1) == key:
            value = match.group(2).strip().strip('"').strip("'")
            if value == DEFAULT_SECRET:
                out.append(f"{key}={secret}{ending}")
                continue
        out.append(line)
    return "".join(out)


def enable_bootstrap_admin(text: str) -> str:
    """Ensure an uncommented ``BOOTSTRAP_ADMIN=1`` line is present."""
    newline = _newline(text)
    found = False
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        body, ending = _split_ending(line)
        stripped = body.strip()
        if not stripped.startswith("#") and _BOOTSTRAP_ASSIGNMENT.match(stripped):
            out.append(f"BOOTSTRAP_ADMIN=1{ending or newline}")
            found = True
            continue
        out.append(line)
    if found:
        return "".join(out)
    if text and not text.endswith(("\n", "\r")):
        out.append(newline)
    out.append(f"BOOTSTRAP_ADMIN=1{newline}")
    return "".join(out)


def clear_bootstrap_admin(text: str) -> str:
    """Remove ``BOOTSTRAP_ADMIN=`` lines, including a commented default."""
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        body, _ending = _split_ending(line)
        if _BOOTSTRAP_ASSIGNMENT.match(body.strip()):
            continue
        kept.append(line)
    return "".join(kept)


def _copy_if_missing(src: Path, dest: Path, root: Path, notes: list[str]) -> None:
    if dest.exists() or not src.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    notes.append(f"created {dest.relative_to(root)}")


def _write(path: Path, text: str) -> None:
    # newline="" keeps the line endings already in ``text`` on Windows.
    path.write_text(text, encoding="utf-8", newline="")


def prepare(
    root: Path,
    *,
    bootstrap_admin: bool = False,
    clear_bootstrap: bool = False,
    secret_factory: Callable[[], str] | None = None,
) -> list[str]:
    """Copy missing env files and fill default secrets under ``root``.

    Returns notes describing what changed. When both flags are set,
    ``clear_bootstrap`` wins.
    """
    new_secret = secret_factory or (lambda: secrets.token_hex(32))
    notes: list[str] = []
    root = root.resolve()

    env_file = root / ".env"
    _copy_if_missing(root / ".env.example", env_file, root, notes)
    if env_file.is_file():
        original = env_file.read_text(encoding="utf-8")
        updated = fill_default_secret(original, "JWT_SECRET", new_secret())
        if updated != original:
            notes.append("set JWT_SECRET")
        elif "JWT_SECRET=" in original:
            notes.append("left existing JWT_SECRET unchanged")
        if clear_bootstrap:
            cleared = clear_bootstrap_admin(updated)
            if cleared != updated:
                notes.append("removed BOOTSTRAP_ADMIN")
            updated = cleared
        elif bootstrap_admin:
            enabled = enable_bootstrap_admin(updated)
            if enabled != updated:
                notes.append("set BOOTSTRAP_ADMIN=1")
            updated = enabled
        if updated != original:
            _write(env_file, updated)

    ui_env = root / "ui" / ".env.local"
    _copy_if_missing(root / "ui" / ".env.local.example", ui_env, root, notes)
    if ui_env.is_file():
        original = ui_env.read_text(encoding="utf-8")
        updated = fill_default_secret(original, "AUTH_SECRET", new_secret())
        if updated != original:
            _write(ui_env, updated)
            notes.append("set AUTH_SECRET")
        elif "AUTH_SECRET=" in original:
            notes.append("left existing AUTH_SECRET unchanged")

    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: directory above this script)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--bootstrap-admin",
        action="store_true",
        help="add BOOTSTRAP_ADMIN=1 so the first admin can register",
    )
    group.add_argument(
        "--clear-bootstrap-admin",
        action="store_true",
        help="remove BOOTSTRAP_ADMIN lines after the admin account exists",
    )
    args = parser.parse_args(argv)
    notes = prepare(
        args.root,
        bootstrap_admin=args.bootstrap_admin,
        clear_bootstrap=args.clear_bootstrap_admin,
    )
    if not notes:
        print("Nothing to change. .env and ui/.env.local are already set.")
        return 0
    for note in notes:
        print(note)
    print("Fill in LLM_API_KEY and EMBEDDING_API_KEY in .env before starting the server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
