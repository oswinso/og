import pathlib
from datetime import timedelta

import matplotlib.collections as mcollections
from matplotlib.animation import FuncAnimation
from rich.progress import Progress, ProgressColumn
from rich.text import Text


class CustomTimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: "Task") -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        delta = timedelta(seconds=delta.seconds, milliseconds=round(delta.microseconds // 1000))
        delta_str = str(delta)
        return Text(delta_str, style="progress.elapsed")


class MutablePatchCollection(mcollections.PatchCollection):
    def __init__(self, patches, *args, **kwargs):
        self._paths = None
        self.patches = patches
        mcollections.PatchCollection.__init__(self, patches, *args, **kwargs)

    def get_paths(self):
        self.set_paths(self.patches)
        return self._paths


def save_anim(ani: FuncAnimation, path: pathlib.Path, **kwargs):
    pbar = Progress(*Progress.get_default_columns(), CustomTimeElapsedColumn())
    pbar.start()
    task = pbar.add_task("Animating", total=ani._save_count)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani.save(path, progress_callback=progress_callback, **kwargs)
    pbar.stop()
