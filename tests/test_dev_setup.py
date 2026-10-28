"""Cross-platform env setup (scripts/dev_setup.py)."""

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("dev_setup", _ROOT / "scripts" / "dev_setup.py")
dev_setup = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(dev_setup)

_SECRET = "ab" * 32  # 64 hex-ish chars; stand-in for token_hex(32)


def _factory() -> str:
    return _SECRET


def test_fill_default_secret_replaces_only_the_placeholder():
    text = (
        "APP_ENV=development\n"
        "JWT_SECRET=change-me-in-production\n"
        "OTHER=change-me-in-production\n"
    )
    out = dev_setup.fill_default_secret(text, "JWT_SECRET", _SECRET)
    assert out == (
        "APP_ENV=development\n"
        f"JWT_SECRET={_SECRET}\n"
        "OTHER=change-me-in-production\n"
    )


def test_fill_default_secret_leaves_a_custom_value():
    text = "JWT_SECRET=already-set-by-the-user\n"
    assert dev_setup.fill_default_secret(text, "JWT_SECRET", _SECRET) == text


def test_fill_default_secret_keeps_crlf():
    text = "JWT_SECRET=change-me-in-production\r\nAPP_ENV=development\r\n"
    out = dev_setup.fill_default_secret(text, "JWT_SECRET", _SECRET)
    assert out == f"JWT_SECRET={_SECRET}\r\nAPP_ENV=development\r\n"


def test_bootstrap_admin_set_and_clear():
    text = "APP_ENV=development\n# Keep this comment\n# BOOTSTRAP_ADMIN=0\n"
    enabled = dev_setup.enable_bootstrap_admin(text)
    assert "# Keep this comment\n" in enabled
    assert "# BOOTSTRAP_ADMIN=0\n" in enabled
    assert enabled.endswith("BOOTSTRAP_ADMIN=1\n")

    again = dev_setup.enable_bootstrap_admin(enabled)
    assert again == enabled

    flipped = dev_setup.enable_bootstrap_admin("BOOTSTRAP_ADMIN=0\n")
    assert flipped == "BOOTSTRAP_ADMIN=1\n"

    cleared = dev_setup.clear_bootstrap_admin(enabled)
    assert "BOOTSTRAP_ADMIN" not in cleared
    assert cleared == "APP_ENV=development\n# Keep this comment\n"


def test_prepare_copies_examples_and_fills_secrets(tmp_path: Path):
    (tmp_path / ".env.example").write_text(
        "APP_ENV=development\nJWT_SECRET=change-me-in-production\n# BOOTSTRAP_ADMIN=0\n",
        encoding="utf-8",
    )
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / ".env.local.example").write_text(
        "SMRITIKOSH_API_URL=http://localhost:8080\nAUTH_SECRET=change-me-in-production\n",
        encoding="utf-8",
    )

    notes = dev_setup.prepare(tmp_path, secret_factory=_factory)

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    ui_env = (ui / ".env.local").read_text(encoding="utf-8")
    assert f"JWT_SECRET={_SECRET}" in env
    assert "JWT_SECRET=change-me-in-production" not in env
    assert "APP_ENV=development" in env
    assert f"AUTH_SECRET={_SECRET}" in ui_env
    assert "change-me-in-production" not in ui_env
    assert "set JWT_SECRET" in notes
    assert "set AUTH_SECRET" in notes


def test_prepare_leaves_existing_secrets_and_toggles_bootstrap(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "JWT_SECRET=custom-secret-value\n# BOOTSTRAP_ADMIN=0\n",
        encoding="utf-8",
    )
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / ".env.local").write_text("AUTH_SECRET=custom-ui-secret\n", encoding="utf-8")

    dev_setup.prepare(tmp_path, bootstrap_admin=True, secret_factory=_factory)
    enabled = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "JWT_SECRET=custom-secret-value" in enabled
    assert _SECRET not in enabled
    assert enabled.rstrip("\n").endswith("BOOTSTRAP_ADMIN=1")
    assert (ui / ".env.local").read_text(encoding="utf-8") == "AUTH_SECRET=custom-ui-secret\n"

    dev_setup.prepare(tmp_path, clear_bootstrap=True, secret_factory=_factory)
    cleared = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BOOTSTRAP_ADMIN" not in cleared
    assert "JWT_SECRET=custom-secret-value\n" in cleared


def test_prepare_uses_real_example_files(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / ".env.example").write_text(
        (repo / ".env.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / ".env.local.example").write_text(
        (repo / "ui" / ".env.local.example").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    dev_setup.prepare(tmp_path, secret_factory=_factory)

    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert f"JWT_SECRET={_SECRET}" in env
    assert "JWT_SECRET=change-me-in-production" not in env
    assert "APP_ENV=development" in env
    ui_env = (ui / ".env.local").read_text(encoding="utf-8")
    assert re.search(rf"^AUTH_SECRET={_SECRET}$", ui_env, re.MULTILINE)
    assert "AUTH_SECRET=change-me-in-production" not in ui_env


def test_main_bootstrap_admin(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / ".env.example").write_text(
        "JWT_SECRET=change-me-in-production\n",
        encoding="utf-8",
    )
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / ".env.local.example").write_text(
        "AUTH_SECRET=change-me-in-production\n",
        encoding="utf-8",
    )

    code = dev_setup.main(["--root", str(tmp_path), "--bootstrap-admin"])
    assert code == 0
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    match = re.search(r"^JWT_SECRET=([0-9a-f]{64})$", env, re.MULTILINE)
    assert match
    assert "BOOTSTRAP_ADMIN=1" in env
    captured = capsys.readouterr()
    assert "set JWT_SECRET" in captured.out
    assert "set BOOTSTRAP_ADMIN=1" in captured.out
