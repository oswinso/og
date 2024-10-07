import pathlib
import subprocess


def save_git_diff(cwd: pathlib.Path, patch_path: pathlib.Path):
    """Save the output of git diff, run at cwd, to patch_path."""
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with patch_path.open("w") as f:
        output = subprocess.run(["git", "diff", "HEAD"], cwd=cwd, capture_output=True, text=True).stdout
        f.write(output)
