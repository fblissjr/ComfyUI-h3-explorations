"""The ordered-reference model, and the one place labels are assigned.

**Pure stdlib on purpose.** Both a runtime node and two static checks have to
agree about what a reference list means, and the checks run with no ComfyUI on
the path. Nothing here imports torch, comfy or this pack's node modules, so
`bench/` can import it directly rather than reconstructing the rule -- which is
how two label authorities drift apart.

## Two models, and why both remain visible

The native node takes references through fixed sockets: `ref_image_0..2`,
`ref_video_0..2`, `ref_video_audio_0..2`, `ref_audio_0..2`. Core walks images,
then videos, then standalone audio, and pairs a soundtrack to a video **by
socket-name suffix**. Order is a property of which socket you plugged into, and
"standalone audio comes last" is not a choice anyone made -- it falls out of
the iteration order.

The ordered model replaces both of those deliberately, and
`docs/research/conditioning_nodes.md` records which contracts are PRESERVED and
which are intentionally REPLACED. Preserved: one shared `<Audio j>` counter, a
soundtrack's label emitted immediately before its video's, a sounded video
producing two presentation items but one DiT block. Replaced: pairing becomes
**ownership** -- a video record carries its own soundtrack, so a mis-numbered
socket cannot silently attach the wrong track -- and position becomes
**list order**, so a standalone audio reference can precede a video.

**A test asserting AGREE on all seven contracts would reject a correct
replacement for doing its job.** `legacy_order` exists so equivalence can be
demonstrated where it is claimed: feed it a socket-shaped input and this module
reproduces the legacy labels exactly, which is the AGREE half. The DIFFER half
is any ordering the socket model could not express.

## Compatibility after migration

Shipped graphs use the ordered model as of 2026-08-23. The legacy adapter stays
because native ComfyUI and historical/hand-built graphs still use sockets, and
because its equivalence checks keep the core behaviour explicit. Label
assignment remains shared; there is no second legacy label authority.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageRef:
    """One still, labelled `<Picture N>`. Keyframes count as images here."""
    name: str = ""


@dataclass(frozen=True)
class VideoRef:
    """One clip, labelled `<Video N>`, which OWNS its optional soundtrack.

    `has_soundtrack` is the whole difference from the socket model. There, a
    track belonged to a video because their socket numbers matched; here it
    belongs because the record says so, and there is no numbering to mismatch.

    `soundtrack_origin` is `None` with no track, `"owned"` when traced to this
    record's own frame loader, `"foreign"` when traced to a different known
    audio source, and `"unresolved"` when the walk stopped at an unfamiliar
    processor.

    **`"foreign"` is accepted, not refused, and that was a decision.** An
    earlier version raised on it. Ownership is established by placing the
    soundtrack inside this record, so provenance is DIAGNOSTIC rather than
    authorization -- and the strict rule was incoherent besides: a raw foreign
    loader was refused while routing the same audio through one unknown
    processor became `"unresolved"` and passed, penalising transparent graphs
    without reliably stopping cross-source audio. Scoring a clip with another
    take's audio is a thing people do on purpose.
    """
    name: str = ""
    has_soundtrack: bool = False
    soundtrack_origin: str | None = None


@dataclass(frozen=True)
class AudioRef:
    """A standalone audio reference, labelled `<Audio N>` from the shared counter."""
    name: str = ""


def assign_labels(records) -> list[str]:
    """The labels the tokenizer will emit, in emission order.

    Three rules, and all three are preserved from core rather than invented:

    - `<Picture N>` and `<Video N>` count independently, from 1.
    - `<Audio N>` is **one shared counter** across soundtracks and standalone
      audio. A prompt saying `<Audio 1>` means something different depending
      on whether a video with sound precedes it, which is why the counter
      cannot be split.
    - A soundtrack's `<Audio j>` is emitted **immediately before** its own
      `<Video k>`, never after.

    What is NOT preserved is that images lead and standalone audio trails.
    Records are emitted in list order, so the caller's order is the answer.
    """
    labels: list[str] = []
    n_img = n_vid = n_aud = 0
    for rec in records:
        if isinstance(rec, ImageRef):
            n_img += 1
            labels.append(f"<Picture {n_img}>")
        elif isinstance(rec, VideoRef):
            if rec.has_soundtrack:
                n_aud += 1
                labels.append(f"<Audio {n_aud}>")
            n_vid += 1
            labels.append(f"<Video {n_vid}>")
        elif isinstance(rec, AudioRef):
            n_aud += 1
            labels.append(f"<Audio {n_aud}>")
        else:
            raise TypeError(f"not a reference record: {rec!r}")
    return labels


def legacy_plan(inputs) -> list:
    """Core's grouped sockets -> the ordered plan they actually mean.

    **The authority here is core's observed behaviour, not
    `check_ref_prompt_labels.wired_labels`.** That distinction is not
    pedantic: `wired_labels` reduces soundtracks to a COUNT and pairs
    `k < n_vaud`, so it cannot represent a sparse socket set, and it gets one
    wrong. Wiring `ref_video_0`, `ref_video_1` and only `ref_video_audio_1`,
    core emits `video, audio, video` -- the track belongs to the SECOND clip,
    by suffix. `wired_labels` reports the audio against the first. Measured
    2026-08-22 by driving core; the ordered model is built against core.

    So this adapter pairs by **exact suffix**, the way
    `comfy_extras/nodes_minimax_h3.py:313-314` does -- `ref_video_audio_N`
    belongs to `ref_video_N` through a string join, and nothing else. It then
    preserves core's group order: images, then videos each carrying whatever
    track its own suffix names, then standalone audio.

    That is the whole legacy model expressed as a plan, which is what makes
    equivalence checkable against core rather than against another
    reimplementation of core.

    **One boundary, narrower than it first looked.** Sockets are read in
    numeric suffix order where core iterates `(ref_images or {}).values()` --
    dict order. Autogrow rebuilds those nested dicts in schema order before
    core sees them, so the sort predicts real execution for hand-built API
    graphs too, not only generated ones. The two diverge only when core is
    called directly in Python with an already-nested, non-canonical dict:
    driven that way on 2026-08-22 with `{"ref_video_1": ..., "ref_video_0":
    ...}` and a track on 0, core emits `video, audio, video` where this says
    `audio, video, video`. That path exists in test harnesses and nowhere
    else, and it is recorded so nobody rediscovers it as a bug in the plan.
    """
    def _wired(prefix):
        return sorted(
            (k for k, v in inputs.items()
             if k.startswith(prefix) and v is not None),
            key=lambda k: int(k.rsplit("_", 1)[-1]))

    records: list = []
    # A keyframe is a <Picture N> as much as a reference is, and core emits
    # them from the plain images= list before any reference.
    for key in ("first_frame", "last_frame"):
        if inputs.get(key) is not None:
            records.append(ImageRef(name=key))
    records += [ImageRef(name=k) for k in _wired("ref_images.ref_image_")]

    tracks = {k.rsplit("_", 1)[-1] for k in
              _wired("ref_video_audios.ref_video_audio_")}
    for k in _wired("ref_videos.ref_video_"):
        suffix = k.rsplit("_", 1)[-1]
        records.append(VideoRef(name=k, has_soundtrack=suffix in tracks))

    records += [AudioRef(name=k) for k in _wired("ref_audios.ref_audio_")]
    return records


def plan_kinds(records) -> list[str]:
    """The record sequence as core's own `ref_items` type strings.

    Core appends TWO presentation entries for a sounded video -- the
    soundtrack's, then the video's -- and one DiT block. Comparing this
    against core's `ref_items` is a stronger equivalence than comparing
    labels, because it checks the sequence rather than a rendering of it.
    """
    kinds: list[str] = []
    for rec in records:
        if isinstance(rec, ImageRef):
            kinds.append("image")
        elif isinstance(rec, VideoRef):
            if rec.has_soundtrack:
                kinds.append("audio")
            kinds.append("video")
        elif isinstance(rec, AudioRef):
            kinds.append("audio")
        else:
            raise TypeError(f"not a reference record: {rec!r}")
    return kinds


def plan_blocks(records) -> list[str]:
    """The record sequence as core's `ref_blocks` kind strings.

    **The other half of contract 1, and asserting only the presentation half
    would miss it.** `ref_items` and `ref_blocks` are deliberately different
    lengths: a sounded video contributes TWO presentation entries -- the
    soundtrack's label, then the video's -- and ONE DiT payload block, tagged
    `video_audio`. Driven through core, one image, two videos with a track on
    the second, and one standalone audio give five items against four blocks.

    A plan that got the labels right and the payload wrong would satisfy
    `plan_kinds` completely, which is why this exists as a separate function
    checked against a separate core list rather than being inferred from the
    first. Added 2026-08-22 after a self-audit found `plan_kinds` was the only
    thing asserted.
    """
    kinds: list[str] = []
    for rec in records:
        if isinstance(rec, ImageRef):
            kinds.append("image")
        elif isinstance(rec, VideoRef):
            kinds.append("video_audio" if rec.has_soundtrack else "video")
        elif isinstance(rec, AudioRef):
            kinds.append("audio")
        else:
            raise TypeError(f"not a reference record: {rec!r}")
    return kinds


# The append-node contract. Named here rather than in the node module so the
# static checks can recognise a chain without importing anything from ComfyUI.
APPEND_KINDS = {
    "MiniMaxH3AppendRefImage": "image",
    "MiniMaxH3AppendRefVideo": "video",
    "MiniMaxH3AppendRefAudio": "audio",
}
CHAIN_INPUT = "references"


class ChainError(ValueError):
    """A reference chain that cannot be resolved. Never a partial plan."""


SOCKET_PREFIXES = ("ref_images.", "ref_videos.", "ref_video_audios.",
                   "ref_audios.")


def wired_sockets(inputs) -> list[str]:
    """Socket keys carrying a link, in the order the mapping presents them."""
    return [k for k, v in inputs.items()
            if k.startswith(SOCKET_PREFIXES) and v is not None]


def plan_for(inputs, graph=None) -> list:
    """Choose a model and build the plan, or refuse to guess between them.

    **A node wiring BOTH a chain and sockets is a partial plan in either
    direction**, and until 2026-08-22 the chain silently won: one image
    appended and two sockets wired reported a single `<Picture 1>` and the
    sockets vanished. Dropping records is a renumbering, so this raises rather
    than picking. Found by self-audit after codex flagged the fail-loudly
    contract as the thing to probe.
    """
    typed = CHAIN_INPUT in inputs
    sockets = wired_sockets(inputs)

    # **Both-models first, on key presence alone.** Ordering decides which
    # error a reader gets, and a node carrying a malformed chain AND sockets
    # has two models wired -- that is the root fault. Reporting the link shape
    # first sends them to fix the link, after which they hit this anyway. Both
    # orders refuse; only one says the useful thing first.
    if typed and sockets:
        raise ChainError(
            f"this node wires an ordered chain on `{CHAIN_INPUT}` AND "
            f"{len(sockets)} legacy socket(s) ({sorted(sockets)}). One of the "
            f"two would be silently dropped, and dropping a record renumbers "
            f"every label after it. Wire one model or the other")

    if typed:
        # **Key presence, not link truthiness.** A node that declares
        # `references` is an ordered node, and every degenerate value it can
        # carry -- an empty list, a link with no graph to resolve it against,
        # a non-list -- used to fall through to `legacy_plan` and return the
        # sockets' answer, or `[]`. Both are partial plans wearing a
        # successful return. Found by codex probing the fail-loudly contract.
        if graph is None:
            raise ChainError(
                f"this node declares `{CHAIN_INPUT}` but no graph was supplied "
                f"to resolve it. The chain holds the plan; without it there is "
                f"nothing to fall back TO, only an empty answer")
        link = inputs.get(CHAIN_INPUT)
        if not (isinstance(link, list) and len(link) == 2):
            raise ChainError(
                f"`{CHAIN_INPUT}` is {link!r}, not a [node_id, slot] link. An "
                f"ordered node with an unresolvable chain has no plan, and "
                f"returning an empty one would silently drop every record")
        if _exact_slot(link[1]) != 0:
            raise ChainError(
                f"`{CHAIN_INPUT}` takes output slot {link[1]} of node "
                f"{link[0]!r}; an append node's plan is slot 0. A different "
                f"slot is a different value and cannot be a chain")

    if typed:
        return resolve_chain(graph, str(inputs[CHAIN_INPUT][0]))
    return legacy_plan(inputs)


def resolve_chain_entries(graph: dict, node_id: str) -> list[tuple[str, dict, str]]:
    """Validate an append chain and return ``(id, node, kind)`` in user order.

    Static consumers that need media links as well as labels use this function
    instead of growing a second chain walker.  Validation therefore remains
    identical whether the caller is assigning labels, pricing references, or
    discovering source files.
    """
    chain, seen = [], set()
    current = node_id
    while current is not None:
        if current in seen:
            raise ChainError(
                f"reference chain revisits node {current!r}: a cycle cannot be "
                f"ordered, and resolving it partially would renumber every "
                f"label after the loop")
        seen.add(current)
        node = graph.get(str(current))
        if node is None:
            raise ChainError(f"reference chain names node {current!r}, "
                             f"which is not in the graph")
        kind = APPEND_KINDS.get(node.get("class_type"))
        if kind is None:
            raise ChainError(
                f"node {current!r} is a {node.get('class_type')!r}, not one of "
                f"{sorted(APPEND_KINDS)}. A reference chain accepts only "
                f"append nodes; anything else means the graph is wired to "
                f"something that does not produce a plan")
        chain.append((current, node, kind))
        link = node.get("inputs", {}).get(CHAIN_INPUT)
        if link is None:
            current = None
        elif isinstance(link, list) and len(link) == 2:
            # The slot matters INSIDE the chain too, not only at the terminal.
            # An append node's plan is output 0; any other slot is a different
            # value with the same shape, and following it would build a plan
            # out of something that is not one.
            if _exact_slot(link[1]) != 0:
                raise ChainError(
                    f"node {current!r} takes `{CHAIN_INPUT}` from output slot "
                    f"{link[1]} of node {link[0]!r}, where an append node's "
                    f"plan is slot 0")
            current = str(link[0])
        else:
            raise ChainError(
                f"node {current!r} has a `{CHAIN_INPUT}` input that is not a "
                f"[node_id, slot] link: {link!r}")
    ordered = list(reversed(chain))
    # Entries are a public view for static consumers, not a weaker structural
    # parse. Validate record-specific ownership here too so a pricing/reporting
    # caller cannot accept a chain that label assignment would refuse.
    for nid, node, kind in ordered:
        if kind == "video":
            _assert_video_ownership(nid, node.get("inputs", {}), graph)
    return ordered


def resolve_chain(graph: dict, node_id: str) -> list:
    """Walk an append chain backward from its terminal and return it in order.

    Each append node takes the plan so far on `references` and contributes one
    record, so the chain is discovered tail-first and reversed. That reversal
    is the only place list order is established, which is why it has its own
    assertion rather than being trusted as obvious.

    **Fails loudly, never partially.** A resolver that returned what it could
    parse would hand a short plan to the label assigner, and a short plan is a
    renumbering: drop one record and every ordinal after it shifts, silently.
    So a cycle, a link to something that is not an append node, or a video
    whose frames and metadata come from different loaders all raise
    `ChainError`.

    The last of those is worth spelling out. A video record owns its
    soundtrack and its `VHS_VIDEOINFO`, and `loaded_fps` is derived from that
    metadata rather than from a second widget somebody could set
    inconsistently. That only holds if the metadata describes the same decode
    as the frames -- so if `frames` and `video_info` resolve to different
    source nodes, the ownership claim is false and the chain is rejected
    rather than trusted.
    """
    records = []
    for nid, node, kind in resolve_chain_entries(graph, node_id):
        ins = node.get("inputs", {})
        if kind == "image":
            records.append(ImageRef(name=str(nid)))
        elif kind == "audio":
            records.append(AudioRef(name=str(nid)))
        else:
            origin = _assert_video_ownership(nid, ins, graph)
            records.append(VideoRef(
                name=str(nid),
                has_soundtrack=ins.get("soundtrack") is not None,
                soundtrack_origin=origin))
    return records


# VHS video sources admitted by the typed record. Matching a node id and slots
# 0/3 is not sufficient: an unrelated four-output node can have values at both
# positions without either being the frames/VHS_VIDEOINFO pair the record
# claims to own. All four rows were re-read from the served `/object_info` on
# 2026-08-23; each emits IMAGE at 0, AUDIO at 2 and VHS_VIDEOINFO at 3.
VIDEO_SOURCE_CLASSES = frozenset({
    "VHS_LoadVideo",
    "VHS_LoadVideoPath",
    "VHS_LoadVideoFFmpeg",
    "VHS_LoadVideoFFmpegPath",
})
VIDEO_SLOTS = {"frames": 0, "video_info": 3}

# Audio nodes a legacy or hand-built soundtrack may legitimately pass through
# on its way from the loader to a record. Shipped typed graphs now wire VHS
# audio directly and cap it in the compiler, but historical trimmed graphs
# remain valid inputs to this compatibility resolver.
# Where audio comes FROM, by class and output slot. Read off the running
# server's `/object_info` on 2026-08-22, never guessed: `GetVideoComponents`
# puts audio at slot 1 and `VHS_LoadVideo` at slot 2, and `LoadVideo` has no
# AUDIO output at all -- it emits VIDEO. An earlier table listed class names
# only and had `LoadVideo` in it, which would have accepted a VIDEO output as
# a soundtrack. Class names are not enough; a class plus a slot is the value.
AUDIO_SOURCE_SLOTS = {
    # All four VHS video loaders agree: images 0, audio 2, video_info 3.
    # Listing only VHS_LoadVideo would report a graph built on any of the
    # other three as "unresolved" -- a weaker answer for wiring that is just
    # as checkable.
    "VHS_LoadVideo": (2,),
    "VHS_LoadVideoPath": (2,),
    "VHS_LoadVideoFFmpeg": (2,),
    "VHS_LoadVideoFFmpegPath": (2,),
    "GetVideoComponents": (1,),
    "LoadAudio": (0,),
    "VHS_LoadAudio": (0,),
    "VHS_LoadAudioUpload": (0,),
}

# Single-input audio nodes a legacy or hand-built soundtrack may legitimately
# pass through, as (input field to follow, output slots that carry audio).
# `TrimAudioDuration` remains supported even though generated typed graphs now
# cap audio inside the compiler.
# `SplitAudioChannels` carries audio on BOTH outputs, which is why the value
# is a tuple rather than a single slot.
AUDIO_PASSTHROUGH = {
    "TrimAudioDuration": ("audio", (0,)),
    "AudioAdjustVolume": ("audio", (0,)),
    "AudioEqualizer3Band": ("audio", (0,)),
    "SplitAudioChannels": ("audio", (0, 1)),
}

# Multi-input audio nodes need their own rule: treating JoinAudioChannels as a
# one-input pass-through makes provenance depend on whichever channel was
# followed and silently drops the other. The live schema names both inputs and
# one AUDIO output. Both branches are traced; see `_trace_soundtrack`.
AUDIO_BRANCHING = {
    "JoinAudioChannels": (("audio_left", "audio_right"), (0,)),
}



def _exact_slot(value) -> int | None:
    """The slot as a real int, or None if it is anything else.

    `int(slot)` was the check until codex probed it, which quietly accepted
    `0.9`, `"0"` and `True` -- every one of them a wrong value with a
    coercible shape, which is the same defect as accepting the right shape
    from the wrong output. `bool` is excluded explicitly because it is an
    `int` subclass and `True == 1` would pass a slot check by accident.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _assert_link(nid, ins, field, required=True):
    """A [node_id, slot] link naming this field's own output slot."""
    link = ins.get(field)
    if link is None:
        if required:
            raise ChainError(f"video append node {nid!r} wires no {field}")
        return None
    if not (isinstance(link, list) and len(link) == 2):
        raise ChainError(f"video append node {nid!r} has a {field} input that "
                         f"is not a [node_id, slot] link: {link!r}")
    want = VIDEO_SLOTS[field]
    if _exact_slot(link[1]) != want:
        raise ChainError(
            f"video append node {nid!r} takes {field} from output slot "
            f"{link[1]} of node {link[0]!r}, where a loader's {field} is slot "
            f"{want}. Accepting any slot means accepting the wrong value with "
            f"the right shape")
    return link


def _trace_soundtrack(graph, link, trail=frozenset()):
    """Follow a soundtrack back through known pass-throughs, carrying SLOTS.

    Returns (known_origin_ids, resolved). The walk tracks `(node_id, slot)`
    pairs, not node ids: an earlier version dropped the slot and called
    `[loader, 0]` -- a VHS_LoadVideo's IMAGE output -- an owned soundtrack,
    along with a fractional slot, a TrimAudioDuration output 7, and a trim
    reading the loader's images instead of its audio. All five reported
    "owned". Class names alone cannot decide this; a class plus a slot is the
    value.

    Raises rather than reporting unresolved when the graph is demonstrably
    malformed -- a missing node, a known class used at a slot that carries no
    audio, a known pass-through with a broken input link, a cycle. Those are
    not "we could not check"; they are "this cannot be right". Unresolved is
    reserved for an unfamiliar node used at a plausible output.
    """
    node_id, slot = str(link[0]), link[1]
    exact = _exact_slot(slot)
    point = (node_id, exact)
    if point in trail:
        raise ChainError(
            f"the soundtrack chain revisits node {node_id!r} slot {slot!r}: "
            f"a cycle has no origin to attribute the audio to")
    trail = trail | {point}
    node = graph.get(node_id)
    if node is None:
        raise ChainError(
            f"the soundtrack traces to node {node_id!r}, which is not in "
            f"the graph. An unresolvable link is not an unfamiliar one")
    cls = node.get("class_type")
    if exact is None or exact < 0:
        raise ChainError(
            f"the soundtrack takes output {slot!r} of node {node_id!r}, "
            f"which is not a real non-negative integer slot")
    if cls in AUDIO_SOURCE_SLOTS:
        if exact not in AUDIO_SOURCE_SLOTS[cls]:
            raise ChainError(
                f"the soundtrack takes output slot {exact} of {cls} "
                f"{node_id!r}, which carries "
                f"{AUDIO_SOURCE_SLOTS[cls]} as audio. That output is not "
                f"a soundtrack whatever its shape")
        return {node_id}, True
    if cls in AUDIO_PASSTHROUGH:
        field, outs = AUDIO_PASSTHROUGH[cls]
        if exact not in outs:
            raise ChainError(
                f"the soundtrack takes output slot {exact} of {cls} "
                f"{node_id!r}, which carries audio on {outs}")
        nxt = node.get("inputs", {}).get(field)
        if not (isinstance(nxt, list) and len(nxt) == 2):
            raise ChainError(
                f"{cls} {node_id!r} has a `{field}` input that is not a "
                f"[node_id, slot] link: {nxt!r}. A known processor with a "
                f"broken input is malformed, not unfamiliar")
        return _trace_soundtrack(graph, nxt, trail)
    if cls in AUDIO_BRANCHING:
        fields, outs = AUDIO_BRANCHING[cls]
        if exact not in outs:
            raise ChainError(
                f"the soundtrack takes output slot {exact} of {cls} "
                f"{node_id!r}, which carries audio on {outs}")
        origins, resolved = set(), True
        for field in fields:
            nxt = node.get("inputs", {}).get(field)
            if not (isinstance(nxt, list) and len(nxt) == 2):
                raise ChainError(
                    f"{cls} {node_id!r} has a `{field}` input that is not a "
                    f"[node_id, slot] link: {nxt!r}. A known processor with a "
                    f"broken branch is malformed, not unfamiliar")
            branch_origins, branch_resolved = _trace_soundtrack(
                graph, nxt, trail)
            origins.update(branch_origins)
            resolved = resolved and branch_resolved
        return origins, resolved
    # An unfamiliar node at a plausible output: it may read the right loader
    # through an input this module does not know. The id is diagnostic only;
    # `resolved=False` prevents it being compared as a known origin.
    return {node_id}, False


def _assert_video_ownership(nid, ins, graph) -> str | None:
    """Frames and metadata from ONE loader; the soundtrack traced to it.

    **Two different provenance claims, and conflating them broke the shipped
    wiring.** Frames and `video_info` must come from the same node at their
    own output slots, because `loaded_fps` is derived from that metadata and
    the claim is false if the two describe different decodes. That is strict.

    A soundtrack is a different matter: it is *expected* to be processed --
    every sounded graph here trims it to the generated duration, and the mono
    upmix would be another such node. Requiring it to arrive raw at slot 2
    rejected the entire shipped population. So it is traced back through
    `AUDIO_PASSTHROUGH` and `AUDIO_BRANCHING` instead. Ownership is established
    by placing it in the record; provenance is diagnostic, so another known
    origin is reported as `foreign` rather than refused.

    Returns the origin verdict for the record to carry.
    """
    frames = _assert_link(nid, ins, "frames")
    info = _assert_link(nid, ins, "video_info")
    if str(frames[0]) != str(info[0]):
        raise ChainError(
            f"video append node {nid!r} takes frames from node {frames[0]!r} "
            f"and video_info from node {info[0]!r}. The record claims to own "
            f"one clip and derives loaded_fps from its metadata, which is "
            f"false if they describe different decodes")

    source = graph.get(str(frames[0]))
    if source is None:
        raise ChainError(
            f"video append node {nid!r} names source node {frames[0]!r}, "
            f"which is not in the graph")
    source_class = source.get("class_type")
    if source_class not in VIDEO_SOURCE_CLASSES:
        raise ChainError(
            f"video append node {nid!r} takes frames and video_info from "
            f"{source_class!r} {frames[0]!r}; matching slots 0/3 do not make "
            f"an unrelated node one of {sorted(VIDEO_SOURCE_CLASSES)}")

    track = ins.get("soundtrack")
    if track is None:
        return None
    if not (isinstance(track, list) and len(track) == 2):
        raise ChainError(f"video append node {nid!r} has a soundtrack input "
                         f"that is not a [node_id, slot] link: {track!r}")
    origins, resolved = _trace_soundtrack(graph, track)
    if not resolved:
        return "unresolved"
    return "owned" if origins == {str(frames[0])} else "foreign"
