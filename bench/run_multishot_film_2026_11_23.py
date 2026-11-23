#!/usr/bin/env python3
"""Shoot a diverse 4-shot (23.50s / 564f) H3 Sound-Mixer film on live ComfyUI (:8188).

Exercises the clean-slate `steering.mixer.MixerCompiler` + `P-H3-Image` anchor +
`heylook 2.0.276` (`h3-director-shot-continue` 14-key `HeylookProvenance`) + Pika
`Ref2VA` (`MiniMaxH3ReleaseReferences(end_percent=0.33)`) across a 4-person live
radio-drama broadcast studio scene with zero overindexed film/noir/market crutches:

  Anchor (`anchor_radio_studio_t2i`, seed=8401):
    `P-H3-Image` `T2I` (`4.0 MP` <- `8.0 MP`, 3:2) copied into `$H3_INPUT_DIR`
    as `multishot_radio_studio_anchor_20261123.png` and inspected by `heylook`
    (`model=""` Auto, `preset=""` auto-inheriting `h3-director-shot-continue`).

  Shot 1 (`0.00-5.88s`, seed=8411, `1x BasicGuider`, `cathedral_opera` macro):
    `(S1)` & `(S2)` trade melodramatic opening lines while `operatic_bellow` (`1.15`)
    and `reverb_cathedral` (`1.10`) stack across `description_only` & `soundscape_only`.

  Shot 2 (`5.88-11.75s`, seed=8412, `2x MiniMaxH3EditGuider` + `MiniMaxH3SpatialDirector`
          + `MiniMaxH3SoundCues` + `power_outage_flashlight`):
    `(S3)` speaks under warm booth practicals (`f0..f47`), then at `t = 2.00s` (`f48`)
    the breaker clicks (`48: click @1.10`, `audio_end_percent=0.60`) and a full-frame
    mask triggers `power_outage_flashlight` (`1.30`, `edited_only`, `scale=1.65`)
    with sweeping emergency flashlight beams (`f48..f140`).

  Shot 3 (`11.75-17.63s`, seed=8413, `1x BasicGuider`, `blizzard_gale` macro):
    `<Subject 4>`'s tabletop wind drum goes rogue (`heavy_blizzard=1.10` +
    `wind_tunnel_gale=1.05`), whipping loose script pages across the booth while
    `(S2)` shouts over indoor swirling snow.

  Shot 4 (`17.63-23.50s`, seed=8414, `1x BasicGuider`, 3-strip cross-bus stack):
    Stacks `blacklight_uv_rave` (`1.15`, Look) + `strobe_club_pulse` (`0.95`, Look) +
    `auctioneer_rapid_patter` (`1.05`, Audio) with `peer_ros=True` and `95.5->98.5`
    smooth knee while `(S1)` delivers a rapid-fire sign-off under UV emergency tubes.

Usage:
  "$PY" bench/run_multishot_film_2026_11_23.py --dry-run
  "$PY" bench/run_multishot_film_2026_11_23.py --run
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client, frames  # noqa: E402

COMFY_ROOT = bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

_PACK_PKG = "_h3pack_film"
if _PACK_PKG not in sys.modules:
    _pkg = types.ModuleType(_PACK_PKG)
    _pkg.__path__ = [str(REPO)]
    sys.modules[_PACK_PKG] = _pkg

_hc = importlib.import_module(f"{_PACK_PKG}.heylook_client")
_mixer = importlib.import_module(f"{_PACK_PKG}.steering.mixer")

import h3_config  # noqa: E402
import preflight_graph as pf  # noqa: E402
import prompts  # noqa: E402

MixerCompiler = _mixer.MixerCompiler
ChannelStrip = _mixer.ChannelStrip

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/multishot_film_2026_11_23"
DATA_DIR = REPO / "data" / "multishot_film_2026_11_23"
ANCHOR_INPUT_NAME = "multishot_radio_studio_anchor_20261123.png"
MASTER_NAME = "multishot_radio_studio_master_564f.mp4"

WIDTH = 1152
HEIGHT = 768
LENGTH = 141  # on_both_clocks = 39 + 51*2; 4 * 141 = 564 frames = 23.50s @ 24 fps
BLACKOUT_FRAME = 48  # exact 3-frame audio clock (2.00s)

SEED_ANCHOR = 8401
SEED_SHOT1 = 8411
SEED_SHOT2 = 8412
SEED_SHOT3 = 8413
SEED_SHOT4 = 8414

UNET_FL2VA = h3_config.MODELS["unet_fl2va"]
UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
IMAGE_VAE = h3_config.IMAGE_GRAPH_VAE
LORA_PH3 = h3_config.PH3_IMAGE_LORA
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PH3_SIGMAS = h3_config.PH3_IMAGE_MANUAL_SIGMAS
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

ANCHOR_T2I_PROMPT = (
    "A wide eye-level live-action film still inside an acoustic wood-paneled mid-century radio drama studio illuminated by warm practical tungsten ceiling fixtures, "
    "with perforated oak acoustic tiles along the walls and a glowing red rectangular sign high on the rear left wall reading \"ON AIR\". "
    "Four distinct people stand spaced clearly across the studio room without overlapping: "
    "on the far left behind a timber foley prop table piled with a shallow wooden tray of gravel, leather shoes, and a hand-cranked canvas wind drum, "
    "a deadpan expressionless foley artist in a buttoned slate-grey wool cardigan holds the crank handle calmly; "
    "left-of-center at a tall stainless boom microphone stand, an older seasoned actor with silver-templed hair, horn-rimmed spectacles, and a tailored burgundy tweed waistcoat over a cream dress shirt holds script pages; "
    "at center beside the second boom microphone, an expressive actress in her thirties with chestnut pinned hair and an emerald green silk scarf draped around a charcoal turtleneck turns toward him; "
    "on the right at the third boom microphone, a wide-eyed young man in brass over-ear studio headphones and a cream ribbed knit sweater grips his paper script with both hands. "
    "Crisp 35mm photography at f/4.5 deep focus, rich fabric weave, balanced tungsten interior key and natural warm bounce light across the parquet floor."
)

_SUBJECT_DEFS = (
    "subject_definitions:\n"
    "<Subject 1> is the silver-templed older radio actor left-of-center in <Picture 1>, wearing a tailored burgundy tweed waistcoat over a cream collared shirt and horn-rimmed spectacles.\n"
    "<Subject 2> is the theatrical actress at the center boom microphone in <Picture 1>, wearing an emerald green silk neck scarf over a charcoal turtleneck.\n"
    "<Subject 3> is the wide-eyed young actor at the right boom microphone in <Picture 1>, wearing brass studio headphones and a cream ribbed knit sweater.\n"
    "<Subject 4> is the expressionless deadpan foley artist at the far-left prop table in <Picture 1>, wearing a buttoned slate-grey cardigan beside the gravel tray and wooden wind drum."
)

_RETENTION_ALL = (
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1]): fully_preserved - exact facial structure, silver-templed hair, horn-rimmed spectacles, burgundy tweed waistcoat, and cream shirt carry directly from <Picture 1>.\n"
    "<Subject 2> (appears in [Shot 1]): fully_preserved - exact facial features, pinned chestnut hair, emerald green silk scarf, and charcoal turtleneck carry directly from <Picture 1>.\n"
    "<Subject 3> (appears in [Shot 1]): fully_preserved - exact youthful face, brass over-ear headphones, and cream ribbed knit sweater carry directly from <Picture 1>.\n"
    "<Subject 4> (appears in [Shot 1]): fully_preserved - exact deadpan face, buttoned slate-grey cardigan, and position behind the left timber foley table carry directly from <Picture 1>."
)

SHOT1_PROMPT = (
    f"{_SUBJECT_DEFS}\n\n"
    "summary:\n"
    "[reference generation] Inside the warm broadcast studio of <Picture 1>, <Subject 1> and <Subject 2> trade soaring melodramatic radio lines while <Subject 4> performs deadpan prop foley as <Subject 3> listens silently.\n\n"
    f"{_RETENTION_ALL}\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action deadpan comedy style with balanced practical tungsten studio lighting, natural filmic tonal rolloff, lifelike skin and cloth texture, and deep focus at 35mm f/4.5.\n"
    "[Shot 1] Wide eye-level shot opening directly on the exact four-person mid-century radio-drama broadcast studio blocking from <Picture 1>, where perforated acoustic oak wall panels frame a glowing warm-white monitor console and a rectangular illuminated red \"ON AIR\" box mounted high on the rear left wall above a heavy timber foley prop bench. Soft ceiling globe bounce light spreads evenly across herringbone parquet flooring, tangled black XLR cables coiled beside heavy cast-iron microphone bases, and three chrome condenser boom stands aligned left-to-center-right. "
    "The camera pushes in smoothly with small amplitude at slow speed straight down the center aisle toward the row of three microphones while holding every performer cleanly in frame. "
    "At the far left behind the prop bench stocked with folded cloth, tin pans, porcelain mugs, leather brogues, and a slatted wooden hand-cranked wind cylinder, <Subject 4> in the buttoned slate-grey cardigan stands completely stiff and expressionless with pale calm features, eyes fixed dead ahead past the actors, turning the shallow pine tray of grey crushed gravel slowly back and forth with one gloved hand, lips sealed firmly throughout, producing no vocal sound. "
    "On the far right beside his upright boom stand, <Subject 3> in the brass over-ear studio headphones and cream ribbed knit sweater clutches his dog-eared script with both pale knuckles tightly against his chest in rigid silence, eyes snapping anxiously left and right toward his two co-stars without parting his lips, producing no vocal sound. "
    "Left of center beneath crisp tungsten rim light, <Subject 1> in the burgundy tweed waistcoat, cream collared shirt, and horn-rimmed spectacles bursts forward onto his toes, plants his left heel cleanly beside the weighted mic base, thrusts his right fist dramatically upward near his silver temple, leans close to the braided wire microphone mesh, and proclaims loudly with puffed chest resonance (S1): <d>[English] Hark! The cellar bolts are giving way tonight!</d> "
    "Closing his mouth promptly into a solemn theatrical glare as he lowers both curled script sheets neatly back to sternum height, <Subject 1> freezes in classic radio poise. "
    "Without missing a beat at center, <Subject 2> in the charcoal wool turtleneck and vivid emerald green silk scarf gathers one flowing scarf tail across her wrist, lifts her chin proudly under the key lamp with luminous expression, and projects in a soaring resonant dramatic alto (S2): <d>[English] Then seal the vault before the tide comes in!</d> "
    "Shutting her lips cleanly after the final word while holding her free palm raised in dramatic suspense toward <Subject 1>, she maintains steady eye contact while <Subject 4> calmly lifts a heavy solid brass padlock and drops it flat onto the gravel tray with deadpan stillness until the continuous five-point-eight-eight-second take concludes cleanly on the final frame.\n\n"
    "overall_soundscape:\n"
    "Warm wooden broadcast room acoustics frame the crunching boots and tumbling gravel inside the wooden foley tray, crisp rustling script paper, a sharp heavy brass padlock clanking onto stones, and subtle analog tape hiss from the monitor desk.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

SHOT2_ORIG_PROMPT = (
    f"{_SUBJECT_DEFS}\n\n"
    "summary:\n"
    "[reference generation] Framed inside the warmly illuminated studio of <Picture 1>, <Subject 3> begins a trembling microphone line under full golden tungsten ceiling globes while <Subject 1>, <Subject 2>, and <Subject 4> watch quietly.\n\n"
    f"{_RETENTION_ALL}\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action deadpan comedy style illuminated evenly by bright warm overhead studio ceiling fixtures with natural tonal rolloff and crisp f/4.5 focus across all surfaces.\n"
    "[Shot 1] Medium-wide three-quarter shot inside the broadcast booth of <Picture 1>, with warm practical overhead tungsten ceiling fixtures shining evenly across all four perforated oak corners, brass studio clock, acoustic drapery along the back glass booth window, heavy timber foley table, and crisp glowing red \"ON AIR\" wall sign. "
    "Every wall corner, floor cable coil, and brass clasp stays brightly illuminated and clearly legible in smooth uniform interior key light as the camera pans right with small amplitude at slow speed across the three boom microphone stands to bring <Subject 3> and the center performers into balanced medium framing while <Subject 4> remains clearly visible on the left behind the prop bench. "
    "<Subject 1> stands stiffly beside the left boom stand in his burgundy tweed waistcoat and horn-rimmed spectacles, folding his script flat against one sleeve with lips pressed shut, producing no vocal sound. "
    "<Subject 2> stands tall at center in her emerald green silk scarf and charcoal turtleneck, listening with raised eyebrows and folded hands while keeping her mouth closed throughout, producing no vocal sound. "
    "Behind the far-left table covered with gravel box, wind drum, and brass latch hardware, <Subject 4> in the grey cardigan calmly picks up an unlit brushed-steel cylinder flashlight and taps its cool bezel gently on the oak table edge, deadpan gaze steady and mouth closed, producing no vocal sound. "
    "In clear balanced warm ceiling light at the right microphone, <Subject 3>, on-screen, twenties, wearing the glossy brass over-ear headphones and cream ribbed knit sweater with earnest wide-eyed intensity, raises one shoulder nervously, adjusts the left brass ear cup slightly against his temple, clears his throat cleanly, leans his chin toward the chrome pop-shield mesh, and speaks timidly in a light uncertain tenor (S3): <d>[English] Wait, did someone trip the main studio breaker just now?</d> "
    "Once his final syllable resolves, his lips close nervously while he glances up toward the bright warm ceiling lamps directly above the condenser booms, shifting his weight slowly from foot to foot while holding the curled bottom corner of his script neatly between both thumbs. "
    "Every ceiling fixture continues burning steadily with warm golden fill light across the wood paneling, parquet grain, and all four actors until <Subject 4> rests the steel flashlight barrel flat beside the gravel box through the final beat of the continuous studio shot.\n\n"
    "overall_soundscape:\n"
    "Close padded broadcast booth air carries a gentle forty-hertz tungsten ballast hum, light paper crinkling in gloved palms, the clean metallic tap of an unlit steel flashlight against oak timber, and soft headphone cord brush against a knit sweater.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

SHOT2_EDIT_PROMPT = (
    f"{_SUBJECT_DEFS}\n\n"
    "summary:\n"
    "[reference generation] Inside the studio of <Picture 1>, a sudden electrical blackout plunges the booth into pitch darkness pierced solely by harsh sweeping flashlight beams as <Subject 3> reacts.\n\n"
    f"{_RETENTION_ALL}\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action deadpan comedy style engulfed in a sudden total electrical power blackout cut solely by harsh sweeping emergency flashlight cones at crisp f/4.5 depth.\n"
    "[Shot 1] Medium-wide three-quarter shot inside the broadcast booth of <Picture 1>, where complete sudden electrical power blackout extinguishes every ceiling light and wall lamp into total ink-black shadow, pierced solely by one fierce narrow handheld flashlight spotlight sweeping cleanly across dark perforated acoustic walls and glowing airborne chalk dust motes. "
    "Only a faint dim ember from the standby glass panel glints high on the pitch-black rear wall as the camera pans right with small amplitude at slow speed across the darkened chrome boom microphone stands toward <Subject 3>, while <Subject 4> flicks the glaring emergency torch beam briskly left-to-right from the far-left prop bench. "
    "<Subject 1> freezes beside the left microphone in his burgundy tweed waistcoat, lips firmly shut, as the sharp roving white beam slices briefly across his horn-rimmed glasses, silver hair, and paper script edges against impenetrable shadow, producing no vocal sound. "
    "<Subject 2> holds her upright stance at center in deep darkness with crisp emerald green silk glinting suddenly as the narrow beam rakes past her collarbone and shoulder line before falling back into shadow, her mouth shut throughout, producing no vocal sound. "
    "On the far left behind the invisible corners of the timber foley bench, <Subject 4> in the slate-grey cardigan stands utterly unbothered in silhouetted half-light, swinging the blinding circular torch cone smoothly across the dark room with deadpan wrist precision and tight closed lips, producing no vocal sound. "
    "Caught square inside the blazing flashlight glare at center-right while bright dust motes dance inside the beam beside the steel mesh screen, <Subject 3>, on-screen, twenties, in brass headphones and cream knit sweater with earnest startled tenor voice (S3), touches his brass headband nervously, blinks sharply against the direct white glare, leans toward the dark microphone silhouette, and calls out anxiously: <d>[English] Wait, did someone trip the main studio breaker just now?</d> "
    "As soon as his last word finishes, his mouth closes rapidly while his wide eyes track the stark beam panning across his face, illuminating script typography and gleaming copper headphone yokes cleanly against deep black room shadow through the final frame of the continuous shot.\n\n"
    "overall_soundscape:\n"
    "Sharp wall-breaker relay click echoes once inside the suddenly silent booth as the overhead ballast hum dies completely, followed by nervous paper rustling, a crisp heavy flashlight toggle click, and hushed leather soles shifting on parquet wood.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

SHOT3_PROMPT = (
    f"{_SUBJECT_DEFS}\n\n"
    "summary:\n"
    "[reference generation] Inside the studio of <Picture 1>, <Subject 4> cranks the tabletop wooden wind machine out of control, unleashing a fierce indoor gale and swirling frost flurries across the cast while <Subject 2> shouts over the roar.\n\n"
    f"{_RETENTION_ALL}\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action deadpan comedy style with balanced practical studio illumination and crisp deep focus at 35mm f/4.5 as an indoor mechanical gale rips across the room.\n"
    "[Shot 1] Wide tracking shot gliding slowly leftward with small amplitude inside the wood-paneled radio broadcast studio of <Picture 1>, where bright tungsten fixtures light all acoustic tiles cleanly beneath the glowing red \"ON AIR\" sign mounted high above the far-left foley table. "
    "With monumental deadpan composure on the left edge behind the heavy oak prop bench, <Subject 4> in the buttoned slate-grey cardigan grips the curved cast-iron crank arm of the slatted canvas wooden foley wind drum and rotates it rapidly faster with smooth relentless circular shoulder drive, posture ramrod straight, staring serenely forward with lips tightly shut, producing no vocal sound. "
    "Out of the roaring spinning cylinder erupts a fierce horizontal indoor gale and cold mist blizzard tearing left-to-right straight across all three chrome boom stands: dozens of loose white typed script sheets rip out of palms and cartwheel tumbling through the air, fine swirling frost crystals and chalky snow flurries whip sideways against the dark oak acoustic panels, and coat hems snap horizontally backward across parquet wood. "
    "Left of center braced into the teeth of the blast, <Subject 1> clutches both horn-rimmed spectacles firmly against his nose with his left hand while raising his burgundy tweed waistcoat collar and sheltering his bent brow behind one forearm, spectacles dusted lightly with frost and lips pinched shut tightly, producing no vocal sound. "
    "On the far right leaning hard forward against the rushing air current, <Subject 3> clamps both palms over his brass over-ear headphone cups as swirling cold air whips hair across his forehead and ruffles his cream ribbed sweater sleeves cleanly, jaw locked tight throughout, producing no vocal sound. "
    "Planting both heels squarely at the center boom stand while loose script leaves whip past her emerald green silk scarf streaming horizontally beside her throat and fine indoor snow flakes sting across the charcoal wool of her turtleneck, <Subject 2>, on-screen, thirties, with a strained, wind-chilled defiant alto cutting cleanly above the tempest (S2), cups one glove beside her mouth toward the quivering condenser mesh and shouts loudly: <d>[English] Who wired the storm drum to three phase power!</d> "
    "Shutting her mouth crisply after the final word while snatching a fluttering script page out of midair and ducking slightly beneath the howling frost wind as <Subject 4> cranks effortlessly to the last frame.\n\n"
    "overall_soundscape:\n"
    "Rising canvas wind-machine screech swells into a thunderous howling gale across the wooden room, accompanied by violently whipping paper sheets, metallic microphone-stand rattle, and crisp hissing wind blasts.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

SHOT4_PROMPT = (
    f"{_SUBJECT_DEFS}\n\n"
    "summary:\n"
    "[reference generation] In the finale beat inside <Picture 1>, ultraviolet emergency tubes pulse violet-cyan across fluorescent props as <Subject 1> delivers a breathless rapid-fire auctioneer broadcast sign-off.\n\n"
    f"{_RETENTION_ALL}\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action deadpan comedy style illuminated by vivid electric ultraviolet blacklight tubes and brisk pulsing club strobes across the broadcast studio at f/4.5 depth.\n"
    "[Shot 1] Medium-wide ensemble shot pushing in steadily with small amplitude at medium speed toward <Subject 1> and the three chrome boom microphone stands inside the broadcast room of <Picture 1>, where every overhead studio lamp has shifted into glowing electric purple-indigo ultraviolet blacklight bars firing crisp timed strobe bursts across the darkened acoustic oak walls. "
    "Vivid fluorescent cyan, acid magenta, and electric violet glow leaps sharply off white script papers, silk emerald scarf embroidery, polished brass headphone yokes, painted \"ON AIR\" lettering, and neon-chalk margin markings across every actor's sleeves in rhythmic optical flashes cleanly synchronized across frames. "
    "At center, <Subject 2> stands completely transfixed under flashing indigo strobes with both hands holding her glowing emerald silk scarf up near her throat and wide eyes glued onto <Subject 1>, mouth neatly shut throughout, producing no vocal sound. "
    "On the right at his neon-outlined boom stand, <Subject 3> freezes in wide-eyed silence beside his radiant brass headphones with glowing paper script gripped rigid between stiff elbows, lips held cleanly shut, producing no vocal sound. "
    "At far left behind the foley gravel tray glowing softly like cold violet sand, <Subject 4> in the slate-grey cardigan stands completely motionless with impassive calm, lifting a glowing pale mint-green kitchen egg-timer high beside one shoulder, producing no vocal sound. "
    "Stepping briskly forward right up against the left-center condenser grill under sharp strobe illumination while turning his wristwatch wrist upward in brisk showman rhythm, <Subject 1>, on-screen, sixties, with luminous white cuffs and bright glowing teeth beneath horn-rimmed rims, unleashes a hyper-articulate, breathless, razor-fast auctioneer baritone-tenor patter (S1) sliding seamlessly through every word with zero pause: <d>[English] Sponsors waiting twenty-two-seconds-gone-going-once-twice-thank-you-all-goodnight!</d> "
    "Landing the final crisp syllable cleanly right as his open palm slaps flat down onto the edge of the timber table and his mouth snaps briskly shut into a triumphant tight smile while <Subject 4> reaches calmly over to flick the brass control toggle off at the end of the continuous shot.\n\n"
    "overall_soundscape:\n"
    "Crisp high-voltage fluorescent hum and subtle rhythmic electrical relay clicks underpin rapid finger snaps, a sharp wooden table slap, dry paper flutter, and clean broadcast room tail resonance.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)


def _input_dir() -> Path:
    val = os.environ.get("H3_INPUT_DIR", "").strip()
    if not val:
        raise SystemExit("H3_INPUT_DIR is unset; run `set -a; . ./h3.env; set +a` first.")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"H3_INPUT_DIR does not exist: {p}")
    return p


def _base_loaders(unet_name: str, *, include_image_vae: bool = False) -> dict[str, Any]:
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    if include_image_vae:
        g["101"] = {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}}
    return g


def _wire_pika_chain(g: dict[str, Any], *, unet_id: str = "1", lora_name: str = LORA_PIKA_REF2VA) -> str:
    g.update({
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": [unet_id, 0],
                "lora_name": lora_name,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "19": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["18", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"}},
        "21": {
            "class_type": "MiniMaxH3Sol",
            "inputs": {
                "model": ["58", 0],
                "tau": 1.0,
                "start_percent": 0.2,
                "end_percent": 1.0,
                "min_tokens": 12288,
                "sink_conditioning": "exact_kv_and_all_rows",
                "verbose": True,
                "dense_blocks": "49",
                "quantizer": "rotated",
                "token_routing": "off",
                "rows": "per segment",
                "rows.text_rows": "exact",
                "rows.reference_rows": "exact",
                "rows.audio_rows": "exact",
                "rows.video_reference_rows": "routed",
            },
        },
    })
    return "21"


def _wire_vhs(g: dict[str, Any], *, vhs_id: str, images_ref: list[Any], audio_ref: list[Any], prefix: str) -> None:
    g[vhs_id] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": images_ref,
            "audio": audio_ref,
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": prefix,
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    }


def _build_anchor_graph() -> dict[str, Any]:
    g = _base_loaders(UNET_FL2VA, include_image_vae=True)
    g.update({
        "102": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_PH3,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "103": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["102", 0], "shift_video": 1.0, "shift_audio": 1.0}},
        "104": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["103", 0], "attention": "comfy kitchen attention"}},
        "105": {"class_type": "ManualSigmas", "inputs": {"sigmas": PH3_SIGMAS}},
        "106": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "107": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "3:2", "generate_megapixels": 8.0, "enforce_4mp_floor": True}},
        "108": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["101", 0],
                "prompt": ANCHOR_T2I_PROMPT,
                "width": ["107", 0],
                "height": ["107", 1],
                "canvas": "explicit",
            },
        },
        "109": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["108", 0]}},
        "113": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED_ANCHOR}},
        "110": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["113", 0], "guider": ["109", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["108", 1]},
        },
        "114": {"class_type": "VAEDecode", "inputs": {"samples": ["110", 0], "vae": ["101", 0]}},
        "111": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["114", 0], "target_width": ["107", 2], "target_height": ["107", 3]}},
        "112": {"class_type": "SaveImage", "inputs": {"images": ["111", 0], "filename_prefix": f"{OUT_PREFIX_DIR}/anchor_radio_studio"}},
    })
    return g


def _build_ref2va_shot(
    arm_name: str,
    *,
    prompt_text: str,
    seed: int,
    plan: Any,
    edited_prompt: str | None = None,
    director_events_json: str | None = None,
    edit_scale: float = 1.65,
) -> dict[str, Any]:
    g = _base_loaders(UNET_REF2VA)
    sol_id = _wire_pika_chain(g, lora_name=LORA_PIKA_REF2VA)
    g.update({
        "15": {"class_type": "LoadImage", "inputs": {"image": ANCHOR_INPUT_NAME}},
        "50": {
            "class_type": "MiniMaxH3AppendRefImage",
            "inputs": {
                "image": ["15", 0],
                "size_policy": "tiered",
                "qwen_view": "separate",
                "qwen_view.qwen_short_edge": 512,
            },
        },
        "132": {
            "class_type": "MiniMaxH3EncodeReferences",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "references": ["50", 0],
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "video_policy": "comfy",
                "image_policy": "comfy",
            },
        },
        "5": {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text}},
    })

    head: list[Any] = [sol_id, 0]
    for idx, sn in enumerate(plan.steer_nodes):
        nid = str(115 + idx)
        g[nid] = {"class_type": "MiniMaxH3Steer", "inputs": sn.to_comfy_inputs(head)}
        head = [nid, 0]

    rel_inputs: dict[str, Any] = {
        "model": copy.deepcopy(head),
        "end_percent": float(plan.release_end_percent),
        "clean_level": 0.999,
        "audio_end_percent": 0.60 if director_events_json else 1.0,
        "audio_clean_level": 1.0,
        "video_end_percent": 1.0,
        "guide_end_percent": 1.0,
    }
    g["135"] = {"class_type": "MiniMaxH3ReleaseReferences", "inputs": rel_inputs}
    head = ["135", 0]

    if plan.pass_mode == "2x":
        if not edited_prompt or not director_events_json:
            raise ValueError(f"{arm_name}: 2x plan requires edited_prompt and director_events_json")
        g["131"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": edited_prompt}}
        g["160"] = {
            "class_type": "MiniMaxH3SpatialDirector",
            "inputs": {
                "mode": "point_targets",
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "feather": 0.08,
                "invert_mask": False,
                "events_json": director_events_json,
            },
        }
        g["70"] = {
            "class_type": "MiniMaxH3SoundCues",
            "inputs": {
                "positive": ["5", 0],
                "latent": ["132", 1],
                "audio_vae": ["4", 0],
                "cues": ["160", 1],
                "snap_to_3f": True,
                "floor_dbfs": -28.0,
                "seed": seed,
                "edited_positive": ["131", 0],
            },
        }
        g["26"] = {
            "class_type": "MiniMaxH3Preflight",
            "inputs": {"conditioning": ["70", 0], "samples": ["70", 1], "edited_conditioning": ["70", 2]},
        }
        g["9"] = {
            "class_type": "MiniMaxH3EditGuider",
            "inputs": {
                "model": copy.deepcopy(head),
                "original": ["26", 0],
                "edited": ["26", 1],  # wired below after checking MiniMaxH3Preflight output order!
                "scale": float(edit_scale),
                "first_step": 0,
                "align_positions": False,
                "last_step": 1000,
                "streams": "both",
                "audio_scale": 1.0,
                "mask": ["160", 0],
            },
        }
        # Note: MiniMaxH3Preflight outputs [CONDITIONING(0), LATENT(1), CONDITIONING(2)]? Wait — check!
        # Let's check MiniMaxH3Preflight return outputs right away! In `h3_world_targets_genie_pika_api.json`
        # node 9 read edited=["70", 2] directly and latent_image=["26", 1]! Let's verify safely:
        g["9"]["inputs"]["edited"] = ["70", 2]
    else:
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": copy.deepcopy(head), "conditioning": ["26", 0]}}

    g.update({
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]},
        },
        "30": {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": copy.deepcopy(head), "sigmas": ["60", 0], "clip": ["2", 0], "note": arm_name},
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
    })
    _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm_name}")
    return g


def compile_shot_plans() -> list[tuple[str, dict[str, Any], Any]]:
    mc = MixerCompiler()
    plan1 = mc.compile(macros=("cathedral_opera",), family="ref2va", has_picture_ref=True, base_release_end_percent=0.33)
    blackout_events = json.dumps(
        {
            "targets": [
                {
                    "id": 1,
                    "label": "BLACKOUT",
                    "start": BLACKOUT_FRAME,
                    "end": LENGTH - 1,
                    "ramp": 0,
                    "x": 0.50,
                    "y": 0.50,
                    "radius": 1.15,
                    "cue": "click",
                    "gain": 1.10,
                }
            ]
        }
    )
    plan2 = mc.compile(
        strips=(ChannelStrip(vector="power_outage_flashlight", strength=1.30, envelope_kind="timeline"),),
        family="ref2va",
        has_picture_ref=True,
        shot_envelope="spatial",
        base_release_end_percent=0.33,
    )
    plan3 = mc.compile(macros=("blizzard_gale",), family="ref2va", has_picture_ref=True, base_release_end_percent=0.33)
    plan4 = mc.compile(
        strips=(
            ChannelStrip(vector="blacklight_uv_rave", strength=1.15),
            ChannelStrip(vector="strobe_club_pulse", strength=0.95),
            ChannelStrip(vector="auctioneer_rapid_patter", strength=1.05),
        ),
        family="ref2va",
        has_picture_ref=True,
        base_release_end_percent=0.33,
    )

    shots = [
        (
            "shot1_opera_opening",
            _build_ref2va_shot("shot1_opera_opening", prompt_text=SHOT1_PROMPT, seed=SEED_SHOT1, plan=plan1),
            plan1,
        ),
        (
            "shot2_timed_blackout",
            _build_ref2va_shot(
                "shot2_timed_blackout",
                prompt_text=SHOT2_ORIG_PROMPT,
                edited_prompt=SHOT2_EDIT_PROMPT,
                director_events_json=blackout_events,
                seed=SEED_SHOT2,
                plan=plan2,
                edit_scale=1.65,
            ),
            plan2,
        ),
        (
            "shot3_blizzard_windmachine",
            _build_ref2va_shot("shot3_blizzard_windmachine", prompt_text=SHOT3_PROMPT, seed=SEED_SHOT3, plan=plan3),
            plan3,
        ),
        (
            "shot4_uv_auctioneer_finale",
            _build_ref2va_shot("shot4_uv_auctioneer_finale", prompt_text=SHOT4_PROMPT, seed=SEED_SHOT4, plan=plan4),
            plan4,
        ),
    ]
    return shots


def verify_all(*, base_url: str) -> None:
    obj_info = comfy_client.get_object_info(base_url=base_url, timeout=20.0)
    anchor_graph = _build_anchor_graph()
    shots = compile_shot_plans()
    all_graphs = [("anchor_radio_studio_t2i", anchor_graph), *[(n, g) for n, g, _ in shots]]

    fail_count = 0
    seen_classes: set[str] = set()
    for name, graph in all_graphs:
        if name != "anchor_radio_studio_t2i":
            carried = prompts.carriers(graph)
            if not carried:
                raise RuntimeError(f"{name}: no prompt carriers found")
            for car in carried:
                node = graph[car.node_id]
                findings = pf.grade(node, graph, name)
                for lvl, msg in findings:
                    print(f"  [{name} node={car.node_id}] {lvl}: {msg}")
                    if lvl == "FAIL":
                        fail_count += 1
                if car.node_id == "5":
                    txt = node["inputs"]["prompt"]
                    dd = txt.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
                    print(f"  [{name}] detailed_description words={len(dd.split())}")
                    for line in pf.price(node, graph):
                        print(f"  [price {name}] {line}")

        for nid, node in graph.items():
            ct = node["class_type"]
            seen_classes.add(ct)
            spec = obj_info.get(ct)
            if not isinstance(spec, dict):
                raise RuntimeError(f"{name} node {nid}: class {ct!r} not in /object_info")
            inp = spec.get("input") or {}
            allowed = set((inp.get("required") or {}).keys()) | set((inp.get("optional") or {}).keys())
            if ct == "VHS_VideoCombine":
                fmt = (node.get("inputs") or {}).get("format")
                fmts = (((inp.get("required") or {}).get("format") or [None, {}])[1] or {}).get("formats") or {}
                for item in fmts.get(fmt) or ():
                    if isinstance(item, list) and item:
                        allowed.add(str(item[0]))
            for k in node.get("inputs") or {}:
                if k.split(".", 1)[0] not in allowed:
                    raise RuntimeError(f"{name} node {nid} ({ct}): unexpected input {k!r}")

    for name, _g, plan in shots:
        print(
            f"  [mixer {name}] pass={plan.pass_mode} guider={plan.guider_class} rel_ep={plan.release_end_percent} "
            f"steers={[(n.vector, n.bus, n.strength, n.target, n.streams, n.edit_pass) for n in plan.steer_nodes]}"
        )
    if fail_count:
        raise RuntimeError(f"Preflight found {fail_count} FAIL(s)")
    print(f"Preflight clean across {len(all_graphs)} graphs and {len(seen_classes)} distinct node classes.")


def _run_heylook_continuity_probe(anchor_png: Path) -> dict[str, Any]:
    """Ask live `heylook 2.0.276` (`model=''` Auto, `preset=''` -> `h3-director-shot-continue`)
    to inspect the rendered T2I anchor and record canonical 14-key `HeylookProvenance`."""
    base = _hc.resolve_heylook_base_url()
    b64, media_type = _hc.prepare_heylook_image(anchor_png)
    local_sys_path = REPO / "docs" / "wiki" / "system_prompts" / "h3_director" / "shot_continuation_system_prompt.md"
    local_sys = local_sys_path.read_text(encoding="utf-8")
    canon_slug = _hc.CANONICAL_PRESET_SLUGS["director_shot_continue"]

    user_text = (
        "Attached is <Picture 1> (the radio broadcast studio anchor keyframe). "
        "Author a concise Mode A Stage-direction continuity note verifying cast left-to-right positions, "
        "wardrobe, lighting temperature, and opening beat into Shot 1 (5.88s, 141 frames) where the silver-haired actor "
        "and emerald-scarf actress trade opening lines while the deadpan foley artist drops a padlock into gravel."
    )

    with _hc.make_heylook_client(base) as client:
        model_row = _hc.discover_heylook_model(client, base, "", need=frozenset({"chat", "vision"}))
        preset_fields, matched_preset = _hc.try_inherit_heylook_preset(client, base, canon_slug, model_row)
        eff_sys = local_sys
        sys_source = "local"
        if preset_fields and preset_fields.get("system"):
            eff_sys, sys_source, _ = _hc.compose_preset_system_prompt(
                local_sys, str(preset_fields["system"]), preset_name=matched_preset
            )
            preset_fields = {k: v for k, v in preset_fields.items() if k != "system"}
        body = _hc.build_heylook_request_body(
            model_row=model_row,
            system_prompt=eff_sys,
            user_prompt=user_text,
            prepared_images=[(b64, media_type)],
            preset_fields=preset_fields,
            sampler_mode="heylook / preset default",
        )
        res = _hc.with_heylook_backoff(lambda: _hc.stream_heylook_message(base, body, client=client))
        prov = _hc.build_heylook_provenance(
            requested_model="",
            model_row=model_row,
            preset=matched_preset,
            system_source=sys_source,
            system_prompt=eff_sys,
            request_body=body,
            res=res,
            repair_used=False,
        )
    return {
        "provenance": prov,
        "continuity_note": res.text.strip(),
        "performance": res.performance or {},
    }


def _pick_output_files(entry: dict[str, Any], out_root: Path) -> dict[str, list[str]]:
    items = comfy_client.collect_saved_files(entry)
    out: dict[str, list[str]] = {"mp4": [], "png": []}
    for it in items:
        sub = str(it.get("subfolder") or "").strip("/")
        fn = str(it.get("filename") or "")
        if not fn:
            continue
        rel = f"{sub}/{fn}" if sub else fn
        if not (out_root / rel).exists():
            continue
        if fn.endswith("-audio.mp4") or (fn.endswith(".mp4") and rel not in out["mp4"]):
            if rel not in out["mp4"]:
                out["mp4"].append(rel)
        elif fn.endswith(".png") and not rel.endswith("-audio.mp4"):
            out["png"].append(rel)
    return out


def _make_1fps_strip(video_path: Path, *, stride: int = 24) -> dict[str, Any]:
    w, h, n_frames = frames.probe(str(video_path))
    stem = video_path.with_suffix("")
    writer = frames.OneASecond(stem, (w, h))
    keep = set(range(0, n_frames, stride))
    if n_frames > 0:
        keep.add(n_frames - 1)
    for idx, rgb in enumerate(frames.stream(str(video_path), (w, h), first=0, pix="rgb24")):
        if idx in keep:
            writer.write(rgb.tobytes(), idx)
    code, parts = writer.close()
    if code != 0:
        raise RuntimeError(f"OneASecond failed on {video_path.name} with exit code {code}")
    return {
        "w": w,
        "h": h,
        "frames": n_frames,
        "parts": [{"file": Path(p["file"]).name, "frames": p["frames"], "bytes": p["bytes"]} for p in parts],
    }


def _decode_rgb_frames(video_path: Path, size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    arr = list(frames.stream(str(video_path), size, first=0, pix="rgb24"))
    return np.stack(arr, axis=0) if arr else np.zeros((0, size[1], size[0], 3), dtype=np.uint8)


def _decode_mono_audio(path: Path, sr: int = 24000) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _audio_stats(path: Path) -> dict[str, Any]:
    wav = _decode_mono_audio(path, sr=24000)
    if wav.size == 0:
        return {"samples": 0}
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    spec = np.abs(np.fft.rfft(wav.astype(np.float64)))
    freqs = np.fft.rfftfreq(wav.size, d=1.0 / 24000.0)
    cent = float((freqs * spec).sum() / max(spec.sum(), 1e-12))
    return {
        "duration_s": round(wav.size / 24000.0, 3),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2),
        "spectral_centroid_hz": round(cent, 1),
    }


def _concat_master_mp4(shot_paths: list[Path], out_path: Path) -> None:
    list_file = out_path.parent / "_concat_inputs.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in shot_paths), encoding="utf-8")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "19",
                "-r",
                "24",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(out_path),
            ],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


def _scrub(obj: Any, out_root: Path, in_root: Path) -> Any:
    o_str = str(out_root).rstrip("/")
    i_str = str(in_root).rstrip("/")
    if isinstance(obj, str):
        return obj.replace(o_str, "$H3_OUTPUT_DIR").replace(i_str, "$H3_INPUT_DIR")
    if isinstance(obj, list):
        return [_scrub(v, out_root, in_root) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v, out_root, in_root) for k, v in obj.items()}
    return obj


def run_all(args: argparse.Namespace) -> int:
    in_root = _input_dir()
    out_root = h3_config.output_dir()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    verify_all(base_url=args.url)
    if not args.run:
        print("Dry-run complete; pass --run to render on live ComfyUI (:8188).")
        return 0

    q = comfy_client.get_queue(base_url=args.url)
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"Refusing to queue: ComfyUI queue is non-empty: {q}")

    client_id = f"multishot_film_{uuid.uuid4().hex[:8]}"

    # 1. Render anchor T2I still via P-H3-Image and copy into $H3_INPUT_DIR
    print("\n>>> Submitting [anchor] anchor_radio_studio_t2i ...", flush=True)
    anchor_pid = comfy_client.submit_prompt(_build_anchor_graph(), client_id=client_id, base_url=args.url)
    anchor_hist = comfy_client.wait_for_prompt(anchor_pid, base_url=args.url, poll_interval=2.0, timeout=900.0)
    anchor_dur = comfy_client.execution_duration_s(anchor_hist)
    anchor_files = _pick_output_files(anchor_hist, out_root)
    if not anchor_files["png"]:
        raise RuntimeError(f"Anchor render produced no PNG output: {anchor_files}")
    anchor_out_path = out_root / anchor_files["png"][0]
    anchor_input_path = in_root / ANCHOR_INPUT_NAME
    tmp_copy = anchor_input_path.with_suffix(".tmp.png")
    shutil.copy2(anchor_out_path, tmp_copy)
    tmp_copy.replace(anchor_input_path)
    print(f"<<< Anchor ready in {anchor_dur}s: {anchor_out_path.name} -> $H3_INPUT_DIR/{ANCHOR_INPUT_NAME}", flush=True)

    # 2. Run live heylook 2.0.276 continuity check & log 14-key HeylookProvenance
    print("\n>>> Querying live heylook 2.0.276 (model='' Auto, preset='' -> h3-director-shot-continue) ...", flush=True)
    hl_info = _run_heylook_continuity_probe(anchor_input_path)
    hl_path = DATA_DIR / "heylook_provenance.json"
    hl_path.write_text(json.dumps(_scrub(hl_info, out_root, in_root), indent=2) + "\n", encoding="utf-8")
    print(
        f"<<< Heylook provenance logged: resolved_model={hl_info['provenance']['resolved_model']!r} "
        f"preset={hl_info['provenance']['preset']!r} sha256={hl_info['provenance']['system_prompt_sha256'][:12]}",
        flush=True,
    )

    # 3. Render 4 consecutive 141f Ref2VA shots via shared client_id
    shots = compile_shot_plans()
    shot_records: list[dict[str, Any]] = []
    shot_mp4s: list[Path] = []
    per_shot_metrics: dict[str, Any] = {}

    for shot_name, graph, plan in shots:
        print(f"\n>>> Submitting [shot] {shot_name} ({plan.pass_mode} {plan.guider_class}) ...", flush=True)
        pid = comfy_client.submit_prompt(graph, client_id=client_id, base_url=args.url)
        hist = comfy_client.wait_for_prompt(pid, base_url=args.url, poll_interval=2.0, timeout=1800.0)
        dur_s = comfy_client.execution_duration_s(hist)
        files = _pick_output_files(hist, out_root)
        audio_mp4s = [rel for rel in files["mp4"] if rel.endswith("-audio.mp4")] or files["mp4"]
        if not audio_mp4s:
            raise RuntimeError(f"{shot_name}: no MP4 output produced: {files}")
        vp = out_root / audio_mp4s[0]
        shot_mp4s.append(vp)
        strip_info = _make_1fps_strip(vp, stride=24)

        rgb = _decode_rgb_frames(vp)
        diff = np.abs(rgb[1:].astype(np.float32) - rgb[:-1].astype(np.float32)).mean(axis=(1, 2, 3))
        luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).mean(axis=(1, 2))
        a_stats = _audio_stats(vp)
        m_entry: dict[str, Any] = {
            "execution_seconds": dur_s,
            "frames": int(rgb.shape[0]),
            "duration_s": round(float(rgb.shape[0]) / 24.0, 3),
            "interframe_mae_mean": round(float(diff.mean()), 3),
            "interframe_mae_max": round(float(diff.max()), 3),
            "mean_luma": round(float(luma.mean()), 2),
            "audio": a_stats,
            "strip_parts": strip_info["parts"],
        }
        if shot_name == "shot2_timed_blackout":
            pre = float(luma[:BLACKOUT_FRAME].mean())
            post = float(luma[BLACKOUT_FRAME:].mean())
            m_entry["pre_blackout_luma_f0_f47"] = round(pre, 2)
            m_entry["post_blackout_luma_f48_f140"] = round(post, 2)
            m_entry["luma_drop_f48_delta"] = round(post - pre, 2)
        per_shot_metrics[shot_name] = m_entry

        shot_records.append({
            "shot": shot_name,
            "prompt_id": pid,
            "pass_mode": plan.pass_mode,
            "guider_class": plan.guider_class,
            "release_end_percent": plan.release_end_percent,
            "steers": [{"vector": s.vector, "bus": s.bus, "strength": s.strength, "target": s.target, "streams": s.streams, "edit_pass": s.edit_pass} for s in plan.steer_nodes],
            "mp4": vp.name,
            "execution_seconds": dur_s,
        })
        print(f"<<< Done {shot_name} ({dur_s}s): {vp.name} | mae={m_entry['interframe_mae_mean']} rms={a_stats.get('rms_dbfs')} dBFS", flush=True)

    # 4. Concatenate into 23.50s (564f) master + master .1fps strip
    out_dir = out_root / OUT_PREFIX_DIR
    master_path = out_dir / MASTER_NAME
    _concat_master_mp4(shot_mp4s, master_path)
    master_strip = _make_1fps_strip(master_path, stride=24)
    mw, mh, mf = frames.probe(str(master_path))
    master_audio = _audio_stats(master_path)

    metrics = {
        "anchor": {
            "prompt_id": anchor_pid,
            "execution_seconds": anchor_dur,
            "file": anchor_out_path.name,
            "input_copy": f"$H3_INPUT_DIR/{ANCHOR_INPUT_NAME}",
        },
        "heylook_provenance": hl_info["provenance"],
        "shots": shot_records,
        "per_shot_metrics": per_shot_metrics,
        "master": {
            "file": f"$H3_OUTPUT_DIR/{OUT_PREFIX_DIR}/{MASTER_NAME}",
            "width": mw,
            "height": mh,
            "frames": mf,
            "duration_s": round(mf / 24.0, 3),
            "bytes": master_path.stat().st_size,
            "audio": master_audio,
            "strip_parts": master_strip["parts"],
        },
    }
    metrics_path = DATA_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(_scrub(metrics, out_root, in_root), indent=2) + "\n", encoding="utf-8")
    print(f"\nMaster concatenated: {mw}x{mh} @ {mf}f ({mf/24.0:.2f}s) -> $H3_OUTPUT_DIR/{OUT_PREFIX_DIR}/{MASTER_NAME}")
    print(f"Wrote {metrics_path.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL, help=f"ComfyUI server URL (default {DEFAULT_URL})")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Compile MixerCompiler plans, grade prompts, and verify :8188 schemas")
    mode.add_argument("--run", action="store_true", help="Render anchor + 4 shots, concatenate 23.50s master, and compute metrics")
    args = ap.parse_args()
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
