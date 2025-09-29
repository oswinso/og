import datetime
import os
import pathlib
import pickle
import warnings
from typing import Any

import attrs
import equinox as eqx
import ipdb
import jax
import jax.tree_util as jtu
import numpy as np
import orbax
import orbax.checkpoint as ocp
from attrs import asdict
from flax.training import orbax_utils
from loguru import logger

from og.cfg_utils import Cfg


class EzManager:
    def __init__(self, mngr: ocp.CheckpointManager, ckpt_dir: pathlib.Path):
        self.mngr = mngr
        self.ckpt_dir = ckpt_dir

        # True if environment variable SUPERCLOUD=1 is set.
        self.on_supercloud = os.environ.get("SUPERCLOUD", "0") == "1"

    def wait_until_finished(self):
        return self.mngr.wait_until_finished()

    def save_ez(self, step: int, items: Any):
        if self.on_supercloud:
            path = self.ckpt_dir / "eqx_ckpt_{:08}.eqx".format(step)
            eqx.tree_serialise_leaves(path, items)
        else:
            if isinstance(items, dict):
                return self._save_ez_dict(step, items)

            return self._save_ez(step, items)

    def save_config(self, config: dict):
        with open(self.ckpt_dir / "config.pkl", "wb") as f:
            pickle.dump(config, f)

    def _save_ez_dict(self, step: int, items: dict):
        args = {}

        primitive_types = (int, float, str, bool)

        for k, v in items.items():
            if isinstance(v, Cfg):
                # Call asdict on all instances of Cfg. Save using JSON.
                arg = ocp.args.JsonSave(v.asdict())
            elif isinstance(v, primitive_types):
                # Save as JSON.
                arg = ocp.args.JsonSave(v)
            elif attrs.has(v):
                # Call asdict on all instances of attrs. Save using JSON.
                arg = ocp.args.JsonSave(asdict(v))
            else:
                # Save as PyTree.
                arg = ocp.args.StandardSave(v)

            args[k] = arg

        self.mngr.save(step, args=ocp.args.Composite(**args))
        ckpt_path = self.mngr._get_save_directory(step, self.mngr.directory)
        return ckpt_path

    def _save_ez(self, step: int, items: Any):
        self.mngr.save(step, args=ocp.args.StandardSave(items))
        ckpt_path = self.mngr._get_save_directory(step, self.mngr.directory)
        return ckpt_path

    def get_all_steps(self) -> list[int]:
        if self.is_eqx_ckpt():
            return self.get_all_steps_eqx()
        else:
            return self.get_all_steps_orbax()

    def get_all_steps_eqx(self) -> list[int]:
        # Eqx ckpts are files saved as eqx_ckpt_00044000.eqx
        ckpt_files = self.ckpt_dir.glob("eqx_ckpt_*.eqx")
        # Parse the step from the file name.
        steps = [int(c.name.split("_")[-1].split(".")[0]) for c in ckpt_files]
        # Sort the steps.
        steps.sort()
        return steps

    def get_all_steps_orbax(self) -> list[int]:
        # Orbax ckpts are saved as folders with the step as the name, i.e.,
        # 00000000, 00001000, 00002000, etc.
        ckpt_dirs = self.ckpt_dir.glob("*/")
        steps = []
        for c in ckpt_dirs:
            try:
                steps.append(int(c.name))
            except:
                # Somehow theres's a "00000000.orbax-checkpoint-tmp-0"..
                if "orbax-checkpoint-tmp" in c.name:
                    continue

                raise ValueError("Error parsing step from ckpt dir: {}".format(c.name))

        # Sort the steps.
        steps.sort()
        return steps

    def is_eqx_ckpt(self):
        # The ckpt is saved using eqx if there are .eqx files in self.ckpt_dir
        return any(f.suffix == ".eqx" for f in self.ckpt_dir.iterdir())

    def _load_ez_dict_eqx(self, items: Any, step: int | None = None):
        """Load using eqx."""
        if step is None:
            # Get the latests ckpt.
            ckpts = sorted(self.ckpt_dir.glob("eqx_ckpt_*.eqx"))
            if len(ckpts) == 0:
                raise ValueError("No eqx ckpt found in {}".format(self.ckpt_dir))

            ckpt_path = ckpts[-1]
            step = int(ckpt_path.name.split("_")[-1].split(".")[0])
        else:
            ckpt_path = self.ckpt_dir / "eqx_ckpt_{:08}.eqx".format(step)

        items_loaded = eqx.tree_deserialise_leaves(ckpt_path, items)
        return items_loaded

    def load_ez(self, items: Any, step: int | None = None):
        assert isinstance(items, dict)

        # See if this ckpt was saved using eqx or not due to running on supercloud.
        if self.is_eqx_ckpt():
            return self._load_ez_dict_eqx(items, step=step)

        return self._load_ez_dict(items, step=step)

    def _load_ez_dict(self, items: Any, step: int | None = None):
        args = {}

        primitive_types = (int, float, str, bool)

        for k, v in items.items():
            if isinstance(v, Cfg):
                # Call asdict on all instances of Cfg. Save using JSON.
                arg = ocp.args.JsonRestore(v.asdict())
            elif isinstance(v, primitive_types):
                # Save as JSON.
                arg = ocp.args.JsonRestore(v)
            elif attrs.has(v):
                # Call asdict on all instances of attrs. Save using JSON.
                arg = ocp.args.JsonRestore(asdict(v))
            else:
                # Save as PyTree.
                arg = ocp.args.StandardRestore(v)

            args[k] = arg
        return self.mngr.restore(step=step, args=ocp.args.Composite(**args))


def get_ckpt_manager(
    ckpt_dir: pathlib.Path, item_names: list[str] | None, max_to_keep: int = 100, step_format_fixed_length: int = 5
):
    # Don't print warning everytime we save
    warnings.filterwarnings("ignore", message=".*Skipped cross-host ArrayMetadata validation.*")

    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, step_format_fixed_length=step_format_fixed_length)
    mngr = ocp.CheckpointManager(ckpt_dir.absolute(), item_names=item_names, options=options)
    return EzManager(mngr, ckpt_dir)


def load_cfg_from_ckpt(ckpt_path: pathlib.Path, name: str) -> Cfg:
    handler = ocp.CompositeCheckpointHandler(name)
    ckpter = ocp.Checkpointer(handler)

    restore_dict = {name: ocp.args.JsonRestore()}
    ckpt_dict = ckpter.restore(ckpt_path, ocp.args.Composite(**restore_dict))
    cfg = Cfg.fromdict(ckpt_dict[name])
    return cfg


def load_from_ckpt(ckpt_path: pathlib.Path, item, name: str | None = None):
    if name is None:
        # If the item is saved directly.
        ckpter = ocp.StandardCheckpointer()
        return ckpter.restore(ckpt_path, ocp.args.StandardRestore(item))

    # Otherwise, assume that it is savaed as composite.
    handler = ocp.CompositeCheckpointHandler(name)
    ckpter = ocp.Checkpointer(handler)

    restore_dict = {name: ocp.args.StandardRestore(item)}
    ckpt_dict = ckpter.restore(ckpt_path, ocp.args.Composite(**restore_dict))
    keyleaves2, _ = jtu.tree_flatten_with_path(ckpt_dict[name])
    keyleaves2 = {tuple(get_key_name(k) for k in kk): v for kk, v in keyleaves2}

    # Only check if we have gpu, otherwise we run into a bug.
    if jax.default_backend() == "cpu":
        logger.warning("Skipping shape checks since only cpu, has orbax bug")
    else:
        restore_dict_flat = {name: ocp.args.StandardRestore()}
        ckpt_dict_flat = ckpter.restore(ckpt_path, ocp.args.Composite(**restore_dict_flat))

        keyleaves1, _ = jtu.tree_flatten_with_path(ckpt_dict_flat[name])
        # jtu.tree_map(check_same_shape, keyleaves1, keyleaves2)
        keyleaves1 = {tuple(get_key_name(k) for k in kk): v for kk, v in keyleaves1}

        for key1, v1 in keyleaves1.items():
            v2 = keyleaves2[key1]
            k_name = "/".join([str(s) for s in key1])
            if np.asarray(v1).shape != np.asarray(v2).shape:
                raise ValueError(f"Shape mismatch for {k_name}: {v1.shape} != {v2.shape}")

    return ckpt_dict[name]


def get_key_name(key) -> int | str:
    """Returns the name of a JAX Key."""
    if isinstance(key, jtu.SequenceKey):
        return key.idx
    elif isinstance(key, jtu.DictKey):
        return str(key.key)
    elif isinstance(key, jtu.GetAttrKey):
        return key.name
    elif isinstance(key, jtu.FlattenedIndexKey):
        return key.key
    else:
        raise ValueError(f'Unsupported KeyEntry: {type(key)}: "{key}"')
