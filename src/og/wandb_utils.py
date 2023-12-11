import wandb


def reorder_wandb_name(wandb_name: str = None, num_width: int = 4, max_word_len: int = 5) -> str:
    name_orig = wandb.run.name
    assert name_orig is not None
    name_parts = name_orig.split("-")
    assert len(name_parts) == 3
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
