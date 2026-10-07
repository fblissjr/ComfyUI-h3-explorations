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

LIP_SYNC = ("<Subject 1> (S1) is the main voice on the track and performs it on screen: the jaw drops and "
            "the lips open on the first syllable of every sung or spoken phrase, the mouth shapes each "
            "vowel and closes on each consonant in time with the voice, and the lips rest together, still, "
            "whenever the voice pauses.")
LIPS_STILL = ("<Subject 1> does not speak or sing at any point: the lips rest together, relaxed and still, "
              "from the first frame to the last, and none of the voices on the track belongs to "
              "<Subject 1>.")
EYES = "The eyes blink naturally and hold a steady line of sight"
CAMERA = "The framing and every movement of the camera are the scene's own and do not change."
SOUNDSCAPE = ("overall_soundscape: The quiet ambience of the scene sits underneath, with the soft brush of "
              "clothing as the performer moves.")
MUSIC = "non_diegetic_music: N/A"

#: What the still provides -> that role's text. `shot` is the paragraph before
#: the performance sentences, `performance` the sentences per voice choice,
#: `close` what ends the shot. `unreferenced`, where a role has it, is the
#: sentence that stands where `MOVES` would when there is no `<Video 1>`.
ROLES: dict[str, dict] = {
    GIVES_WHOLE: dict(
        definition=("<Subject 1> is the {who} shown in <Picture 1>, preserving {poss} facial identity, hair, "
                    "build and the clothing visible in <Picture 1>{motion}. The background, lighting and "
                    "framing of <Picture 1> are not present in the target video."),
        summary=("[reference generation] <Subject 1> takes the place of one person in a scene that is "
                 "already lit, framed and cut, {voice}, while everything else in the scene stays as it is."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair, build "
                   "and clothing in every frame, at every distance from the camera and from every side; "
                   "only the setting changes."),
        scene=("The target video is photorealistic live-action, and its setting, its lighting, its framing "
               "and every other person and object in it stay exactly as they already are from the first "
               "frame to the last."),
        place=("[Shot 1] <Subject 1> is in the scene for the whole take, in the place the scene holds for "
               "one person, at that place's distance from the lens and at the scale of everything around "
               "it."),
        shot=(
            "The scene's own light falls on <Subject 1> exactly as it falls on what is beside {obj}: the "
            "same direction, the same softness and the same colour on the face, the hair and the clothing, "
            "and whenever the light changes colour or brightness, the light on <Subject 1> changes with it "
            "at the same moment.",
            "The shadow <Subject 1> casts lies where that light sends it and moves when <Subject 1> moves.",
            "Where the frame shows the whole figure, <Subject 1> is whole, in the clothing of <Picture 1>, "
            "and whatever <Picture 1> does not show of {poss} clothing is plain and in keeping with it; "
            "where the frame shows only the head and shoulders, the face of <Subject 1> fills that space at "
            "the same scale, sharp and evenly exposed, with the skin texture, the hairline and the eyes of "
            "<Picture 1>.",
            "Seen from the side or from behind, <Subject 1> keeps the same hair, the same build and the "
            "same clothing.",
        ),
        performance={
            VOICE_MAIN: (
                LIP_SYNC,
                "A breath lifts the chest and shoulders before each new phrase.",
                EYES + ", and the brows and cheeks carry the feeling of the line.",
                "The head tips and turns with the delivery, the shoulders loosen and move with the rhythm, "
                "and the hands move with the phrasing, opening on a long note and settling as it ends.",
            ),
            VOICE_SILENT: (
                LIPS_STILL,
                "The breathing is slow and even, lifting the chest a little.",
                EYES + ", and the face stays attentive to what is happening in the scene.",
                "The head, the shoulders and the hands move a little and often, the small shifts of a "
                "person at ease, in time with the rhythm of the scene.",
            ),
        },
        close=("Everyone and everything else is untouched: wherever <Subject 1> does not cover them, the "
               "other people, the walls, the furniture and the ground are visible exactly as before, steady "
               "and in focus."),
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
        summary=("[reference generation] <Subject 1> is in a scene that is already lit, framed and cut. "
                 "From the waist up <Subject 1> is as <Picture 1> shows {obj}, {voice}."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - the {who}: {poss} face and "
                   "everything <Picture 1> shows of {obj} from the waist up are retained in every frame."),
        scene="The target video is photorealistic live-action.",
        place="[Shot 1] <Subject 1>, the {who}, is in the scene.",
        unreferenced="The upper body of <Subject 1> and the legs below it move as one body.",
        shot=(
            "The scene's own light falls on <Subject 1> as it falls on everything around {obj}, from the "
            "same direction and in the same colour, and changes on {obj} when it changes there.",
        ),
        performance={
            VOICE_MAIN: ("<Subject 1> (S1) is the main voice heard on the track and performs it as it plays.",),
            VOICE_SILENT: ("<Subject 1> does not speak or sing at any point, and none of the voices on the "
                           "track belongs to <Subject 1>.",),
        },
        close="",
    ),
    GIVES_HEAD: dict(
        definition=("<Subject 1> is the {who} shown in <Picture 1>, preserving {poss} facial identity, "
                    "{poss} hair and anything worn on the head in <Picture 1>{motion}. The clothing, "
                    "background, lighting and framing of <Picture 1> are not present in the target video."),
        summary=("[reference generation] The head of <Subject 1> takes the place of one person's head in a "
                 "scene that is already lit, framed and cut, on that person's own body and in that "
                 "person's own clothes, {voice}, while everything else in the scene stays as it is."),
        retention=("<Subject 1> (appears in [Shot 1]): fully_preserved - retain the same face, hair and "
                   "headwear in every frame, at every distance from the camera and from every side; the "
                   "body, the clothing and the setting are the scene's own."),
        scene=("The target video is photorealistic live-action, and its setting, its lighting, its framing, "
               "every other person and object in it, and the body, the clothing and the movement of the "
               "one person whose head is replaced stay exactly as they already are from the first frame to "
               "the last. Only that person's head changes: it is the head of <Subject 1>."),
        place=("[Shot 1] The head of <Subject 1> sits on that person's neck for the whole take, at the same "
               "size and in the same place, and meets the collar of the clothing with no gap and no second "
               "neckline."),
        shot=(
            "The head always faces the way the body beneath it faces: when the body turns, the head turns "
            "with it at the same moment and by the same amount, and when the body has its back to the "
            "camera the camera sees only the back of the head of <Subject 1> and whatever is worn on it, "
            "never the face.",
            "The hair of <Subject 1> is the only hair: none of the original hair remains on the shoulders, "
            "the chest or the back, where the clothing, its print and its folds carry on unbroken.",
            "The scene's own light falls on the head exactly as it falls on the body below it and on what "
            "is beside it: the same direction, the same softness and the same colour, and whenever the "
            "light changes colour or brightness, the light on the face changes with it at the same moment.",
            "Where the frame is close, the face is sharp and evenly exposed, with the skin texture, the "
            "hairline and the eyes of <Picture 1>.",
        ),
        performance={
            VOICE_MAIN: (
                LIP_SYNC,
                EYES + ", and the head tips and nods with the delivery while the body carries on moving "
                "exactly as before.",
            ),
            VOICE_SILENT: (
                LIPS_STILL,
                EYES + ", and the head moves only as the body beneath it moves.",
            ),
        },
        close=("Everyone and everything else is untouched: the other people, the walls, the furniture and "
               "the ground are visible exactly as before, steady and in focus."),
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
        else "movement: from the prompt (the Masked Source's motion reference is off)" if replace is not None
        else "movement: from the prompt (no Masked Source wired)",
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
    added to the shot as written, after the performance sentences.
    """
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r}; one of {list(VOICES)}")
    role = ROLES[resolve_gives(picture_gives, replace)]
    moving = motion_reference not in (None, MOTION_NONE)
    poss, obj, _word = pronouns(subject)
    fill = dict(poss=poss, obj=obj, motion=MOTION_CLAUSE if moving else "",
                voice=SUMMARY_VOICE[voice], who="{who}", added="{added}")
    moves = [MOVES] if moving else [role["unreferenced"]] if "unreferenced" in role else []
    shot = [role["place"]] + moves + list(role["shot"]) + list(role["performance"][voice])
    added = re.sub(r"\s+", " ", add_to_shot or "").strip()
    if added:
        shot.append("{added}")
    shot += ([role["close"]] if role["close"] else []) + [CAMERA]
    sections = [
        "subject_definitions:\n" + "\n".join([role["definition"]] + ([VIDEO_DEFINITION] if moving else [])),
        "summary:\n" + role["summary"],
        "retention_analysis:\n" + "\n".join([role["retention"]] + ([VIDEO_RETENTION] if moving else [])),
        "detailed_description:\n" + role["scene"] + "\n" + " ".join(shot),
        SOUNDSCAPE,
        MUSIC,
    ]
    # the user's words go in last and literally: they may hold a brace or a `__list__` placeholder
    return "\n\n".join(sections).format(**fill).replace("{who}", who(subject)).replace("{added}", added)
