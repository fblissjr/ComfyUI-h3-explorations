"""The masked video-to-video prompt, assembled from a few choices.

H3's reference format is six sections of long prose, and in the masked lane
most of it never changes: the plate holds the setting, the framing and the
cuts, so the text names none of them and one prompt serves every window of
any clip (owner, 2026-10-04: the lane must not need a prompt written for a
shot). What does change is small: who the reference still shows (a few of
the user's own words, from which the pronouns follow), whether they are the
voice on the track, what of them the still provides, and whether the
original's movement is shown to the model as `<Video 1>`. Until
2026-10-06 each combination was a file somebody typed out, and turning the
motion reference on with the old text left `<Video 1>` unnamed.

`assemble` writes the text from those choices. Every sentence is a constant
below, so a wording change is one edit and every graph that wires the node
(`masked_prompt.py`) picks it up; `bench/check_masked_prompt.py` holds the
output to the bank's copies of the texts that have been rendered.

No torch and no ComfyUI: the node, the generator and `workflows/prompts.py`
(which resolves the node's text for the graders) all import this file.

Where the sentences come from. **Rendered**: the whole-person texts are
mrhf's generic swap prompts of 2026-10-04 (one window, one seed each, on the
band clip) and the `<Video 1>` lines are the ones the motion arms measured
(`bench/results/2026-10-05_masked_v2v_motion_arms.md`). The noun is not
free: on the one pair rendered, the motion text for "the person" turned the
subject as the text for "the man" did and started the turn later
(`bench/results/2026-10-06_masked_v2v_person_text.md`; one window, one
seed), so `subject` is worth setting to match the still. **Not rendered**:
any subject beyond those three words (`blonde haired woman`, `man wearing a
red cap`), the silent variant, and every head-and-hair text, which generalises
`prompt_bank/ref2va_masked_head_swap.txt` (rendered on the band clip) the
way the whole-person text generalised its own first version: no count of
people, no duration, no camera claim. **Rendered as a typed text, not as
this wording**: the head-and-upper-body role (2026-10-06). A text typed for
one still, with that still's clothing named in it, rendered twice on the
ref2va motion graph with Sapiens2's hair, face, upper clothing and hands
wired into the Masked Source, and both were the first renders of that clip's
widest window the owner accepted (the masking board, finding mj-06; one
clip, one seed, two window lengths). The role below is that text with the
still's own words moved into `subject`; its sentences for a render with no
motion reference have not rendered at all.

**Cut on 2026-10-10, and the texts below have not rendered as they now
stand.** The owner's rule of that day is that a text says nothing of a
window that is not known to be so, and that it is a caption of the finished
shot, never an instruction about an edit. An audit of this file
(`bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`) found sentences
of both kinds in every role, and they went: see the note above `ROLES`.
What "rendered" means above is therefore the earlier wording; the
whole-person and head-and-hair shots are about half their old length, and
the head-and-upper-body role, which already said almost none of it and
is the one wording a person has accepted, changed only in its sound line
where a motion reference is wired (its unrendered sentence for a render
with none was cut):
four sentences of the kinds cut elsewhere are held in it until a render
says they can go (`HELD`). `docs/wiki/decisions.md`, 2026-10-10.
"""
from __future__ import annotations

import re

# ---- the choices -----------------------------------------------------------

#: `subject` is the user's own words for who the still shows, and completes
#: "<Subject 1> is the ... shown in <Picture 1>": `person`, `woman`,
#: `blonde haired woman`, `man wearing a red cap`. The three below are the
#: forms that have rendered.
SUBJECT_PERSON = "person"
SUBJECT_MAN = "man"
SUBJECT_WOMAN = "woman"
#: The words that set the pronouns, read off `subject`: the first one found
#: decides, and none means "their". Reasoned, and kept short on purpose: a
#: wrong guess shows in the text the node displays, and the wording fixes it.
HIS = ("man", "boy", "guy", "gentleman", "male")
HER = ("woman", "girl", "lady", "female")

VOICE_MAIN = "the main voice on the track"
VOICE_SILENT = "silent"
VOICES = (VOICE_MAIN, VOICE_SILENT)

GIVES_FOLLOW = "what the Masked Source replaces"
GIVES_WHOLE = "the whole person"
GIVES_UPPER = "the head and upper body"
GIVES_HEAD = "the head and hair"
GIVES = (GIVES_FOLLOW, GIVES_WHOLE, GIVES_UPPER, GIVES_HEAD)

# The Masked Source's own values. COPIES, because `video_mask.py` imports
# torch and ComfyUI and this file must not; `bench/check_masked_prompt.py`
# pins each against the original.
REPLACE_WHOLE = "whole subject"
REPLACE_PART = "head and hair"
REPLACE_PARTS = "the wired parts"
MOTION_NONE = "none"

# ---- the sentences ---------------------------------------------------------
# `{poss}` and `{obj}` are the pronouns `pronouns` reads off the subject;
# `{motion}` and `{voice}` come from the clauses below; `{who}` is the user's
# own words for the subject, put in last and literally.

MOTION_CLAUSE = (", whose body motion, posture, gestures, head movements and their timing come from the "
                 "person in <Video 1>")
VIDEO_DEFINITION = ("<Video 1> is the source of the movement transferred to <Subject 1>; its scene is not "
                    "reused and its person's appearance is not.")
VIDEO_RETENTION = ("<Video 1> (motion source): attribute_transfer - only the body motion, posture, gestures "
                   "and their timing are taken; the scene and the person's appearance are not.")
# The relationship and nothing else (owner, 2026-10-07): a list of what the movements might be
# ("turning when they turn ...") is read as an instruction, and the subject turned.
MOVES = "<Subject 1> moves as the person in <Video 1> moves, at the same moments."

SUMMARY_VOICE = {VOICE_MAIN: "and performs the main voice heard on the track to its timing",
                 VOICE_SILENT: "without speaking or singing"}

# "each phrase of that voice", not "every sung or spoken phrase" (2026-10-10): a track has other voices, and in a
# window where one of them sings the old wording told the subject's mouth to follow it.
LIP_SYNC = ("<Subject 1> (S1) is the main voice on the track and performs it on screen: the jaw drops and "
            "the lips open on the first syllable of each phrase of that voice, the mouth shapes each "
            "vowel and closes on each consonant in time with it, and the lips rest together, still, "
            "whenever that voice pauses.")
LIPS_STILL = ("<Subject 1> does not speak or sing at any point: the lips rest together, relaxed and still, "
              "from the first frame to the last, and none of the voices on the track belongs to "
              "<Subject 1>.")
#: The head-and-upper-body role's four sentences kept as accepted (see the note above `ROLES`): the scene in
#: its summary, the style opening, the light on the subject, and the camera. `bench/check_masked_prompt.py`
#: reads this tuple, so deleting a sentence here and in the role is the whole of the later cut.
HELD_SCENE = "<Subject 1> is in a scene that is already lit, framed and cut."
HELD_OPENING = "The target video is photorealistic live-action."
HELD_LIGHT = ("The scene's own light falls on <Subject 1> as it falls on everything around {obj}, from the "
              "same direction and in the same colour, and changes on {obj} when it changes there.")
HELD_CAMERA = "The framing and every movement of the camera are the scene's own and do not change."
HELD = (HELD_SCENE, HELD_OPENING, HELD_LIGHT, HELD_CAMERA)
#: The one sentence every role's shot opens with: the subject, in the user's words, is there. It says nothing of
#: where, for how long or at what size, which the node cannot know of a window it never sees.
PLACE = "[Shot 1] <Subject 1>, the {who}, is in the scene."
#: The soundscape, in general words, since the node never hears the track. THE SAME SENTENCE the reference
#: composer writes when no reused audio is wired (`workflows/build_workflows.py::_ref_prompt` reads this
#: constant), so the pack has one generic line and not two. `N/A` is not available here: the guide keeps it for
#: a user who asked for complete silence (`vendor_guides/base_en.md`, 4.6). Reasoned, not measured: the field
#: was not tested on its own. It asserts ambience over a track nobody heard, and less than the line it replaced;
#: approved by the lead, read and accepted by the owner, 2026-10-10 (as the lead relays it).
AMBIENCE = "Natural ambient atmosphere continues throughout the shot."
SOUNDSCAPE = "overall_soundscape: " + AMBIENCE
#: `N/A` is the guide's form for "no music that only the audience hears" (`vendor_guides/base_en.md`, 4.7), and
#: the guide counts singing and music the people on screen can hear as part of the description, not of this
#: field. So it holds where the subject performs the track, which is every shipped masked graph. It is NOT known
#: to hold with `voice` silent on a track that has a score nobody on screen makes: the node never hears the
#: track. No generic sentence stands in for it, because any sentence here asserts such a score.
MUSIC = "non_diegetic_music: N/A"

#: "The [clothing,] background, lighting and framing of <Picture 1> are not present in the target video", in two
#: definitions, is OURS and not the guide's form: the guide never says what a reference does not supply. It is
#: kept because it says how much of a WIRED reference is used, which the node knows, and in the head role it is
#: the one sentence that keeps the still's clothing out.
#:
#: What the still provides -> that role's text. `shot` is what follows the place and movement sentences,
#: `performance` the sentences per voice choice. `opening`, where a role has it, is a line before the shot, and
#: `close` a sentence after anything `add_to_shot` adds. With no `<Video 1>` no role says anything of movement:
#: the head-and-upper-body role had a sentence for that case (the upper body and the legs "move as one body"),
#: which asserted legs in the frame and that they move, had never rendered, and was cut with the rest.
#:
#: **What these texts no longer say (owner, 2026-10-10; `docs/wiki/decisions.md`, that date;
#: `bench/results/2026-10-10_prompt_text_hardcoded_in_code.md` has every sentence that went).** Two kinds were
#: cut from all three roles. (1) Anything the node cannot know of a clip it never sees: that the picture is
#: photorealistic live-action, that the subject is there for the whole take at one size, what else stands in
#: the scene, that the plate is steady and sharp, a shadow, the original's hair and a print, and what the
#: subject's breath, eyes, head, shoulders and hands do. (2) Anything that speaks of an edit: what stays, what
#: is kept, what is "the scene's own", "exactly as before", "takes the place of". The model was trained on
#: captions of finished shots and is told of no edit, so each sentence left is a plain statement of what the
#: finished shot shows, or a relationship to a wired reference. No style sentence opens the description: the
#: guide asks for one (`vendor_guides/ref_en.md`, 5.2) and the node cannot write a true one.
#:
#: **The head-and-upper-body role is held back, on purpose (the lead, 2026-10-10).** It is the one wording a
#: person has watched and accepted, and an accepted result is not changed without a render. So it keeps four
#: sentences of the kinds cut elsewhere, exactly as accepted: `HELD` below. They go after one render of the
#: upper graph with and without them on a window already accepted; until then only its sound line changed.
ROLES: dict[str, dict] = {
    GIVES_WHOLE: dict(
        definition=("<Subject 1> is the {who} shown in <Picture 1>, preserving {poss} facial identity, hair, "
                    "build and the clothing visible in <Picture 1>{motion}. The background, lighting and "
                    "framing of <Picture 1> are not present in the target video."),
        summary="[reference generation] <Subject 1> is in the scene {voice}.",
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair, build "
                   "and clothing in every frame, at every distance from the camera and from every side."),
        shot=(
            "Where the frame shows the whole figure, <Subject 1> is whole, in the clothing of <Picture 1>, "
            "and whatever <Picture 1> does not show of {poss} clothing is plain and in keeping with it; "
            "where the frame shows only the head and shoulders, it shows the face of <Subject 1> with the "
            "skin texture, the hairline and the eyes of <Picture 1>.",
            "Seen from the side or from behind, <Subject 1> keeps the same hair, the same build and the "
            "same clothing.",
        ),
        performance={VOICE_MAIN: (LIP_SYNC,), VOICE_SILENT: (LIPS_STILL,)},
    ),
    # A still that shows a person from the chest up cannot dress their legs: asked for the whole
    # person it gave a head at the portrait's scale or nobody, and asked for this it held
    # (2026-10-06, the docstring's last paragraph). The Masked Source's `replace` has no value
    # that means it, so it is chosen here, with the matching parts wired.
    # In the vendor guide's form since 2026-10-07 (owner, on two seeds of one window:
    # bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md, section 4): the finished
    # scene is described, not an edit to it; the subject's own words are said in the definition,
    # the retention line and the shot; nothing says what the subject will do, because the text is
    # written without seeing the clip. One shot paragraph: this node does not know which window
    # of a clip a render takes, so it cannot count that window's cuts.
    GIVES_UPPER: dict(
        definition="<Subject 1> is the {who} shown in <Picture 1>{motion}.",
        summary=("[reference generation] " + HELD_SCENE + " From the waist up <Subject 1> is as "
                 "<Picture 1> shows {obj}, {voice}."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the {who}: {poss} face and "
                   "everything <Picture 1> shows of {obj} from the waist up are retained in every frame."),
        opening=HELD_OPENING,
        shot=(HELD_LIGHT,),
        close=HELD_CAMERA,
        performance={
            VOICE_MAIN: ("<Subject 1> (S1) is the main voice heard on the track and performs it as it plays.",),
            VOICE_SILENT: ("<Subject 1> does not speak or sing at any point, and none of the voices on the "
                           "track belongs to <Subject 1>.",),
        },
    ),
    GIVES_HEAD: dict(
        definition=("<Subject 1> is the {who} shown in <Picture 1>, preserving {poss} facial identity, "
                    "{poss} hair and anything worn on the head in <Picture 1>{motion}. The clothing, "
                    "background, lighting and framing of <Picture 1> are not present in the target video."),
        summary=("[reference generation] <Subject 1> is in the scene, with the face, the hair and anything "
                 "worn on the head as <Picture 1> shows them, {voice}."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair and "
                   "headwear in every frame, at every distance from the camera and from every side."),
        shot=(
            # the one thing kept of the old paragraph about the head and the body: a face must not be drawn
            # on the back of a head. Said as what the shot shows, on a condition, with no "body beneath".
            "When <Subject 1> has {poss} back to the camera, the camera sees the back of {poss} head and "
            "whatever is worn on it, never the face.",
            "Where the frame is close, the face has the skin texture, the hairline and the eyes of "
            "<Picture 1>.",
        ),
        performance={VOICE_MAIN: (LIP_SYNC,), VOICE_SILENT: (LIPS_STILL,)},
    ),
}

def resolve_gives(picture_gives: str, replace: str | None) -> str:
    """What the still provides: the choice itself, or read off the Masked Source's `replace`."""
    if picture_gives in ROLES:
        return picture_gives
    if picture_gives != GIVES_FOLLOW:
        raise ValueError(f"unknown picture_gives {picture_gives!r}; one of {list(GIVES)}")
    if replace is None or replace == REPLACE_WHOLE:
        return GIVES_WHOLE
    if replace == REPLACE_PART:
        return GIVES_HEAD
    raise ValueError(
        f"the Masked Source replaces `{replace}`, which can be any part of the subject: set `picture_gives` "
        f"to what <Picture 1> provides, one of {list(ROLES)}")


def who(subject: str) -> str:
    """The user's words for the subject as they sit in the sentence: one line, no leading article, no full stop."""
    words = re.sub(r"\s+", " ", subject or "").strip().strip(",.;: ")
    words = re.sub(r"^(?:a|an|the)\s+", "", words, flags=re.IGNORECASE)
    return words or SUBJECT_PERSON


def pronouns(subject: str) -> tuple[str, str, str]:
    """(possessive, object pronoun, the word that decided) for a subject; the word is "" when none did."""
    found = [(m.start(), m.group(0).lower()) for m in re.finditer(r"[A-Za-z]+", who(subject))
             if m.group(0).lower() in HIS + HER]
    if not found:
        return "their", "them", ""
    word = min(found)[1]
    return ("his", "him", word) if word in HIS else ("her", "her", word)


def summary(subject: str = SUBJECT_PERSON, voice: str = VOICE_MAIN, picture_gives: str = GIVES_FOLLOW,
            add_to_shot: str = "", replace: str | None = None, motion_reference: str | None = None) -> str:
    """What the node did with each input, one line each, for the user to read above the text."""
    poss, _obj, word = pronouns(subject)
    gives = resolve_gives(picture_gives, replace)
    moving = motion_reference not in (None, MOTION_NONE)
    lines = [
        f"subject: \"the {who(subject)} shown in <Picture 1>\"; pronouns: {poss}"
        + (f" (from \"{word}\")" if word else " (no man or woman word in the subject)"),
        f"voice: {voice}",
        f"the still gives: {gives}"
        + (" (read from the Masked Source's `replace`)" if picture_gives == GIVES_FOLLOW and replace is not None
           else " (no Masked Source wired)" if picture_gives == GIVES_FOLLOW else " (set here)"),
        "movement: from <Video 1>, the Masked Source's motion reference" if moving
        else "movement: the text says nothing of it (the Masked Source's motion reference is off)" if replace is not None
        else "movement: the text says nothing of it (no Masked Source wired)",
        "added to the shot: " + (re.sub(r"\s+", " ", add_to_shot or "").strip() or "nothing"),
    ]
    return "\n".join(lines)


def assemble(subject: str = SUBJECT_PERSON, voice: str = VOICE_MAIN, picture_gives: str = GIVES_FOLLOW,
             add_to_shot: str = "", replace: str | None = None, motion_reference: str | None = None) -> str:
    """The prompt for one masked render.

    `subject` is who the still shows, in the user's words; it sits in the
    subject's definition, and the pronouns follow it (`pronouns`). The text
    outranks the picture where the two disagree (`docs/prompting.md`,
    "Silence is not neutral"), so words that match the still hold a look the
    render drifts from and words that do not match override it. `replace`
    and `motion_reference` are the Masked Source's own values, or None when
    no source is wired (the whole person, no `<Video 1>`). `add_to_shot` is
    added to the shot as written, after the performance sentences; it ends
    the shot in every role but the head-and-upper-body one, whose camera
    sentence still follows it.
    """
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r}; one of {list(VOICES)}")
    role = ROLES[resolve_gives(picture_gives, replace)]
    moving = motion_reference not in (None, MOTION_NONE)
    poss, obj, _word = pronouns(subject)
    fill = dict(poss=poss, obj=obj, motion=MOTION_CLAUSE if moving else "",
                voice=SUMMARY_VOICE[voice], who="{who}", added="{added}")
    moves = [MOVES] if moving else []
    shot = [PLACE] + moves + list(role["shot"]) + list(role["performance"][voice])
    added = re.sub(r"\s+", " ", add_to_shot or "").strip()
    if added:
        shot.append("{added}")
    shot += [role["close"]] if "close" in role else []
    opening = role["opening"] + "\n" if "opening" in role else ""
    sections = [
        "subject_definitions:\n" + "\n".join([role["definition"]] + ([VIDEO_DEFINITION] if moving else [])),
        "summary:\n" + role["summary"],
        "retention_analysis:\n" + "\n".join([role["retention"]] + ([VIDEO_RETENTION] if moving else [])),
        "detailed_description:\n" + opening + " ".join(shot),
        SOUNDSCAPE,
        MUSIC,
    ]
    # the user's words go in last and literally: they may hold a brace or a `__list__` placeholder
    return "\n\n".join(sections).format(**fill).replace("{who}", who(subject)).replace("{added}", added)
