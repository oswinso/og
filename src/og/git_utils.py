import pathlib
import subprocess


def save_git_diff(cwd: pathlib.Path, patch_path: pathlib.Path):
    """Save the output of git diff --cached, run at cwd, to patch_path."""
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with patch_path.open("w") as f:
        f.write(subprocess.run(["git", "diff", "--cached"], cwd=cwd, capture_output=True, text=True).stdout)
