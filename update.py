"""
Git auto-update helpers for the yuuka-comfyui custom nodes.

The logic mirrors the character-gallery project's update flow: it checks
the upstream branch for new commits, pulls them when available, and
surface messages back to the caller so the host application can decide
what to do next.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent

UPDATE_STATUS = {
    "UP_TO_DATE": 0,
    "AHEAD": 1,
    "ERROR": -1,
    "SKIP": 2,
}

_AUTO_UPDATE_RAN = False


def _log(message: str) -> None:
    print(f"[Yuuka Auto Update] {message}")


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def _run_git_command(command: list[str]) -> Tuple[Optional[str], Optional[str]]:
    """Run a git command anchored at the repository root."""
    try:
        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(REPO_ROOT),
            startupinfo=startupinfo,
        )
        return result.stdout.strip(), None
    except FileNotFoundError:
        return None, "Git executable not found. Please install Git and make sure it is on PATH."
    except subprocess.CalledProcessError as exc:
        return None, exc.stderr.strip() or exc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def check_for_updates() -> Tuple[int, str, bool]:
    """
    Return a tuple (status, message, dependencies_changed).

    dependencies_changed indicates whether requirements.txt or any plugin.json
    file changed between local HEAD and the upstream tracking branch.
    """
    if not _is_git_repo():
        return UPDATE_STATUS["SKIP"], "Git repository not detected; auto-update skipped.", False

    _, error = _run_git_command(["git", "fetch"])
    if error:
        return UPDATE_STATUS["ERROR"], f"Failed to fetch from remote: {error}", False

    local_hash, error = _run_git_command(["git", "rev-parse", "HEAD"])
    if error:
        return UPDATE_STATUS["ERROR"], f"Cannot determine local HEAD: {error}", False

    remote_hash, error = _run_git_command(["git", "rev-parse", "@{u}"])
    if error:
        return UPDATE_STATUS["ERROR"], (
            "Cannot determine upstream commit. Is this branch tracking a remote? "
            f"Git error: {error}"
        ), False

    if local_hash == remote_hash:
        return UPDATE_STATUS["UP_TO_DATE"], "Repository already up to date.", False

    changed_files_str, error = _run_git_command(["git", "diff", "--name-only", f"{local_hash}..{remote_hash}"])
    if error:
        return UPDATE_STATUS["AHEAD"], "Update detected (changed files could not be listed).", True

    changed_files = [line.strip() for line in changed_files_str.splitlines() if line.strip()]
    dependencies_changed = "requirements.txt" in changed_files or any(name.endswith("plugin.json") for name in changed_files)

    return UPDATE_STATUS["AHEAD"], "Update detected on remote branch.", dependencies_changed


def perform_update() -> bool:
    """Execute `git pull --ff-only` to fast-forward the repository."""
    _log("Attempting to fast-forward repository with 'git pull --ff-only'.")
    output, error = _run_git_command(["git", "pull", "--ff-only"])
    if error:
        _log(f"Failed to apply update automatically: {error}")
        return False

    if output:
        _log(f"Pull output:\n{output}")
    else:
        _log("Repository already up to date after pull.")
    return True


def auto_update_if_needed(force: bool = False) -> bool:
    """
    Check for updates once per interpreter session and apply them if available.

    Returns True when an update was applied, False otherwise.
    """
    global _AUTO_UPDATE_RAN  # noqa: PLW0603
    if _AUTO_UPDATE_RAN and not force:
        return False
    _AUTO_UPDATE_RAN = True

    env_value = os.getenv("YUUKA_COMFYUI_AUTO_UPDATE", "1").strip().lower()
    if env_value in {"0", "false", "off"}:
        _log("Auto-update disabled via YUUKA_COMFYUI_AUTO_UPDATE environment variable.")
        return False

    status, message, dependencies_changed = check_for_updates()

    if status == UPDATE_STATUS["ERROR"]:
        _log(message)
        return False

    if status == UPDATE_STATUS["SKIP"]:
        _log(message)
        return False

    if status == UPDATE_STATUS["UP_TO_DATE"]:
        # Keep logs quiet when there is nothing to do.
        return False

    _log(message)
    if not perform_update():
        return False

    _log("Repository updated successfully. Please restart ComfyUI to load the new code.")

    if dependencies_changed and (REPO_ROOT / "requirements.txt").exists():
        _log(
            "Detected dependency changes. "
            "Run 'pip install -r requirements.txt' if additional packages are required."
        )

    return True


__all__ = [
    "UPDATE_STATUS",
    "auto_update_if_needed",
    "check_for_updates",
    "perform_update",
]
