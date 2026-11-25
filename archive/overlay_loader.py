"""Load a research checkpoint as an overlay on the released one, a piece at a time.

`checkpoint_overlay.py` says what an overlay holds. This node reads the base
the overlay names from `diffusion_models`, applies the pieces the widgets
keep, and hands the state dict to core's `load_diffusion_model_state_dict`, so
core's model detection builds whatever the keys describe (the gate modules
when the gates are kept) exactly as it would from a file. Holding the base in
memory costs what a normal load costs.

Selection: the backbone diff of the blocks named, then each of the gates, the
token refiner, adaln with its time table, and the tensors outside the blocks.
A piece the overlay does not carry selects nothing. `gate_scale` multiplies
every gate row scale (`bench/results/2026-09-29_fasth3_overlay_size.md`).
"""

from __future__ import annotations

import os

import folder_paths
from comfy_api.latest import io

import comfy.sd
import comfy.utils

from .block_spec import parse_blocks
from .checkpoint_overlay import apply_overlay, check_base, read_overlay, select_pieces

OVERLAY_FOLDER = "h3_overlays"
OVERLAY_SUFFIX = ".h3overlay.safetensors"


# Registered at import, before any schema reads the folder's file list.
folder_paths.add_model_folder_path(OVERLAY_FOLDER, os.path.join(folder_paths.models_dir, OVERLAY_FOLDER), is_default=True)
folder_paths.folder_names_and_paths[OVERLAY_FOLDER] = (folder_paths.folder_names_and_paths[OVERLAY_FOLDER][0], {".safetensors"})


class MiniMaxH3OverlayLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3OverlayLoader",
            display_name="MiniMax H3 Overlay Loader",
            category="MiniMax H3/loaders",
            description=(
                "Loads a research checkpoint stored as an overlay on the released one it was built "
                "on (models/h3_overlays), applying only the pieces selected: the backbone diff of "
                "some blocks, the gates, the token refiner, adaln with its time table, and the "
                "tensors outside the blocks. The result is core's own load of the assembled state "
                "dict. Replaces Load Diffusion Model for these files."),
            inputs=[
                io.Combo.Input("overlay_name", options=[n for n in folder_paths.get_filename_list(OVERLAY_FOLDER) if n.endswith(OVERLAY_SUFFIX)]),
                io.String.Input("blocks", default="all",
                                tooltip="'all', or DiT block indices and ranges such as '0-24,40'. Whose backbone "
                                        "diff (codes, row scales, norms) applies."),
                io.Boolean.Input("gates", default=True, tooltip="The overlay's to_gate_compress tensors."),
                io.Boolean.Input("refiner", default=True, tooltip="The overlay's token refiner tensors."),
                io.Boolean.Input("adaln", default=True, tooltip="The overlay's adaln tensors and time table; they apply together."),
                io.Boolean.Input("io_layers", default=True, tooltip="The overlay's patch and condition projections and final layer."),
                io.Float.Input("gate_scale", default=1.0, min=0.0, max=10.0, step=0.01,
                               tooltip="Multiplies every gate row scale. 1.0 leaves them as stored; anything else needs the gates."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, overlay_name, blocks="all", gates=True, refiner=True, adaln=True, io_layers=True, gate_scale=1.0) -> io.NodeOutput:
        overlay_path = folder_paths.get_full_path_or_raise(OVERLAY_FOLDER, overlay_name)
        meta, pieces = read_overlay(overlay_path)
        base_path = folder_paths.get_full_path_or_raise("diffusion_models", meta["base_name"])
        check_base(meta, base_path)
        sd, metadata = comfy.utils.load_torch_file(base_path, return_metadata=True)
        count = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        chosen = frozenset(range(count)) if blocks.strip() == "all" else parse_blocks(blocks, count)
        sd = apply_overlay(sd, overlay_path, select_pieces(pieces, chosen, gates, refiner, adaln, io_layers), gate_scale)
        model = comfy.sd.load_diffusion_model_state_dict(sd, metadata=metadata)
        if model is None:
            raise RuntimeError(f"core could not detect a model from {overlay_name} on {meta['base_name']}")
        return io.NodeOutput(model)
