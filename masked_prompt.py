"""The node that writes the masked video-to-video prompt.

`masked_prompt_text.py` holds every sentence and says where each came from;
this is the node on top of it. Wire `prompt` into the song node's `prompt`
and the Masked Source's `source` into this node: the text then names the
motion reference exactly when the Masked Source builds one, and describes
what it replaces, so the two cannot disagree (until 2026-10-06 turning the
motion reference on with an older text left `<Video 1>` unnamed, and nothing
said so).

**Four things to set, and what the node works out itself.** `subject` is
who the still shows, in the user's words, and is the one input most renders
need: it completes a sentence the tooltip shows. The pronouns are read off
it. `voice` and `picture_gives` are choices with a default that fits most
clips, and `picture_gives` reads the Masked Source unless told otherwise.
`add_to_shot` is free text for one render. Whether the text names
`<Video 1>` is never set here: it follows the Masked Source. Above the text,
the node shows one line per input saying what it did with it, so nothing it
worked out is hidden.

**What the `source` wire costs.** The song node's `preview` asks for no
loader and no model, but it does ask for its prompt, and this node asks for
the Masked Source. So with `source` wired, a preview tracks the subject.
(Until 2026-10-10 the Masked Source kept that mask on disk and the render
that followed tracked nothing; that store is removed. Whether the render
tracks again is now core's own node cache's to say, not this pack's.)
Reading the two settings from the queued graph instead would avoid that and
was not done: core computes a node's cache fingerprint without the graph
(`execution.py`, the `fingerprint_inputs` call passes no prompt), so a
changed setting on the Masked Source would have left this node's old text in
the cache.

To change one sentence for every render, edit the constant in
`masked_prompt_text.py`; to write the whole text by hand, type it into the
song node's `prompt` as before and leave this node out.
"""
from __future__ import annotations

import logging

from comfy_api.latest import io, ui

from . import masked_prompt_text as text
from .part_coverage import record_line
from .video_mask import H3MaskedSource

logger = logging.getLogger(__name__)


class MiniMaxH3MaskedPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedPrompt",
            display_name="MiniMax H3 Masked Prompt (video to video)",
            category="model/latent/minimax",
            description=(
                "Writes the prompt for a masked video-to-video render, so the long reference-format text is "
                "never typed. Say who the still shows in `subject`; the rest has defaults. Wire `prompt` into "
                "the song node's `prompt` and the Masked Source's `source` into this node. The node shows what "
                "it did with each input, then the text."),
            inputs=[
                io.String.Input("subject", default=text.SUBJECT_PERSON,
                                tooltip=("Who the still shows, in a few words. It completes the sentence "
                                         "\"<Subject 1> is the ... shown in <Picture 1>\".\n\n"
                                         "Examples: `person`, `woman`, `blonde haired woman`, `man wearing a "
                                         "red cap`.\n\n"
                                         "Say only what is in the still: where the words and the still "
                                         "disagree, the words win. `man` or `woman` in it sets the pronouns.")),
                io.Combo.Input("voice", options=list(text.VOICES), default=text.VOICE_MAIN,
                               tooltip=("Whether the new subject is the one heard on the track.\n\n"
                                        "`the main voice on the track`: they sing or speak it, lips in time.\n\n"
                                        "`silent`: their lips stay closed.")),
                io.Combo.Input("picture_gives", options=list(text.GIVES), default=text.GIVES_FOLLOW,
                               tooltip=("What the model takes from the still.\n\n"
                                        "`what the Masked Source replaces`: worked out from the wired `source`. "
                                        "Leave it here unless the Masked Source replaces `the wired parts`.\n\n"
                                        "`the whole person`: face, hair, build and clothing.\n\n"
                                        "`the head and upper body`: face, hair, headwear and the clothing "
                                        "above the waist, on the original's legs. For a still that shows "
                                        "the person from the chest up; wire the matching parts into the "
                                        "Masked Source.\n\n"
                                        "`the head and hair`: the head only, on the original's body and "
                                        "clothes.")),
                io.String.Input("add_to_shot", multiline=True, default="",
                                tooltip=("Optional. Sentences added to the description of the shot, as written. "
                                         "Call the subject <Subject 1>.\n\n"
                                         "Example: `<Subject 1> moves in step with the people on either side.`")),
                H3MaskedSource.Input("source", optional=True,
                                     tooltip=("The Masked Source's `source` output. The text then follows it: "
                                              "what is replaced, and the motion reference when it is on.")),
            ],
            outputs=[io.String.Output(display_name="prompt", tooltip="The text, for the song node's `prompt`.")],
        )

    @classmethod
    def execute(cls, subject=text.SUBJECT_PERSON, voice=text.VOICE_MAIN, picture_gives=text.GIVES_FOLLOW,
                add_to_shot="", source=None) -> io.NodeOutput:
        replace = source.get("replace") if source is not None else None
        motion = source.get("motion_reference") if source is not None else None
        prompt = text.assemble(subject, voice, picture_gives, add_to_shot, replace, motion)
        did = text.summary(subject, voice, picture_gives, add_to_shot, replace, motion)
        # The Masked Source's warning about the part mask, shown and logged here because a preview runs this
        # node and not the song node's source. It is not part of the prompt: `prompt` is the same with it.
        warned = record_line(source)
        if warned:
            did = f"{did}\n{warned}"
        if source is not None and source.get("motion_frames_2") is not None:
            # said, since an unnamed reference is one the model cannot know to use; the text itself is not changed
            did = (f"{did}\na second motion video is wired to the Masked Source: this node writes nothing about <Video 2>. "
                   "Say what it is and what is taken from it in `add_to_shot`, or write the prompt by hand")
        logger.info("[h3] MiniMaxH3MaskedPrompt: %s", did.replace("\n", "; "))
        return io.NodeOutput(prompt, ui=ui.PreviewText(f"{did}\n\n{prompt}"))
