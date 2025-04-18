import datetime

import wandb
from loguru import logger


def reorder_wandb_name(wandb_name: str = None, num_width: int = 4, max_word_len: int = 5) -> str:
    name_orig = wandb.run.name
    if name_orig == "" or name_orig is None:
        # Probably offline. Generate a new name.
        if wandb_name is not None:
            return wandb_name

        stamp_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        name = f"offline-run-{stamp_str}"
        logger.info("Offline, so using name `{}`".format(name))
        wandb.run.name = name
        return name

    assert name_orig is not None
    name_parts = name_orig.split("-")

    if name_parts[0] == "dummy":
        # For wandb disabled.
        return wandb.run.name

    # They changed the name format to maybe have 3 words (dazzling-candy-heart-14).
    # In this case, just keep the first two.
    if len(name_parts) > 3:
        name_parts = [name_parts[0], name_parts[1], name_parts[-1]]
    # assert len(name_parts) == 3
    word0, word1, num = name_parts
    # If words are too long, then truncate them.
    word0, word1 = word0[:max_word_len], word1[:max_word_len]
    num = num.zfill(num_width)
    if wandb_name is not None:
        new_name = "{}-{}".format(num, wandb_name)
    else:
        new_name = "{}-{}-{}".format(num, word0, word1)
    wandb.run.name = new_name
    return new_name


def flatten_dict(d: dict, separator: str = ".") -> dict:
    """Flatten a arbitrarily nested dictionary. Used since wandb's online viz doesn't work well with nested dicts."""

    def _flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{separator}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten_dict(d)
