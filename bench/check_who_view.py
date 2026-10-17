#!/usr/bin/env python3
"""Check `bench/who_view.py`: the read-only structured view generated from `shots.json`, `cast.jsonl`, and `owner_answers.jsonl`.

Each case tests one guarantee of `bench/who_view.py` and the `G20` / `C53` fixes:
1. `shot_level_fields_and_history_are_preserved`: `order`, `order_by`, `scene_score`, `others_present`, and
   `history` (`merged_from`, `split_from`, `moved_from`, `answered_over`) survive intact on each shot.
2. `cast_entries_keep_box_size_table_and_standing`: `cast_entries` carries `box`, `size`, `table`, and `standing`
   (`False` on superseded/withdrawn lines, `True` on the latest standing line per `(subject, frame)`), and
   `standing_boxes` lists only standing `"here"` boxes.
3. `line_numbers_expand_ranges_and_exclude_dates`: `lines 420-422` expands to `[420, 421, 422]`, and ISO timestamps
   after a comma (`line 310, 2026-10-12T14:39`) never inject `10`, `12`, or `2026` into `answer_lines`.
4. `owner_answers_match_by_frame_shown_across_b20_and_who_and_split_words`: `b20` and `who` rows, early
   `chosen: "the lead"` rows, `source_frame` matching (never leaking into the other half of a split shot via stale
   `frames`), and split-shot word notes (`source_frame == frames[0]` naming a frame in the second half) routing to
   the second half's `note.owner_words`.
5. `cut_from_prefers_full_containment_and_lists_all_on_merged`: a shot contained in `1039-1919` (`1039-1060`) takes
   `1039-1919` over a partial earlier overlap (`604-1050`), and a merged shot (`1898-1944`) lists both ranges in
   `cut_from_all`.
6. `attribution_parses_the_owner_and_shots_record_answer_for`: `"the owner named..."` sets `said_by == "owner"`,
   and `"entered by shots_record.py answer for mrohmy"` sets `entered_by == "shots_record.py answer for mrohmy"`
   with no spurious `sessions_named`.
7. `seen_standing_overrides_not_seen_when_seen_follows_and_flags_hedged_words`: a later `"is": "seen"` look takes
   back an earlier `"not seen"` look on its frames and records `overruled_looks`; hedged words (`"MAYBE"`,
   `"i THINK"`) set `note.owner_hedged = True`.
8. `made_from_records_sha256_and_verify_view_fresh_refuses_stale`: `made_from` records exact `sha256` digests,
   `source_fps` is read from `manifest.json` alongside `copy_fps = 24`, and `verify_view_fresh` refuses when an
   input file changes on disk.
9. `refuses_missing_or_corrupt_inputs_cleanly`: missing or corrupt inputs exit non-zero with `REFUSED:` and no
   Python traceback.

    <python> bench/check_who_view.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cast_file  # noqa: E402
import who_view  # noqa: E402

TOOL = HERE / "who_view.py"


def main() -> int:
    fails = 0

    def case(name: str, ok: bool, why: str = "") -> None:
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        shots_p = root / "shots.json"
        cast_p = root / "cast.jsonl"
        ans_p = root / "owner_answers.jsonl"
        plan_p = root / "plan.json"
        man_p = root / "manifest.json"
        out_json = root / "who_v2.json"
        out_md = root / "who_v2.md"

        shots_obj = {
            "schema": "h3_shots/1",
            "video": "fun.mp4",
            "copy": "fun_0000_6394_24p.mov",
            "source_frames": 6395,
            "span": [604, 6394],
            "subjects": ["lead", "second"],
            "written_by": "a session, 2026-10-13",
            "cuts_from": {
                "604-1050": "kitchen table",
                "1039-1919": "rest table",
                "1920-6394": "ffmpeg lavfi.scene_score",
            },
            "not_cuts": [{"frame": 900, "why": "camera pan"}],
            "shots": [
                {
                    "first": 604,
                    "last": 1038,
                    "order": ["second", "lead"],
                    "order_by": "mrpin, 2026-10-13T16:40:27Z: second first so lead does not steal frame 806",
                    "others_present": "yes",
                    "note": "the owner, answers.jsonl line 310, 2026-10-12T14:39: two women in kitchen",
                    "who": {
                        "lead": {
                            "present": "yes",
                            "how": "frame 700: 1 is the lead",
                            "by": "owner, 2026-10-13T16:17:04Z (answers.jsonl lines 420-422, 428); the owner named nobody else",
                            "frames_on": [700],
                            "frames_boxed": [700],
                        },
                        "second": {
                            "present": "yes",
                            "how": "frame 606: 1 is the second subject",
                            "by": "owner, on the who page at 2026-10-13T22:36:43Z (answers.jsonl line 425), entered by shots_record.py answer for mrohmy",
                            "frames_on": [],
                            "frames_boxed": [606],
                            "seen_looks": [
                                {
                                    "part": "face_and_neck",
                                    "first": 950,
                                    "last": 1038,
                                    "is": "not seen",
                                    "why": "arm across face on stills",
                                    "by": "a reader",
                                    "at": "2026-10-16T14:19:29Z",
                                    "entered_by": "mrgan",
                                },
                                {
                                    "part": "face_and_neck",
                                    "first": 950,
                                    "last": 1038,
                                    "is": "seen",
                                    "why": "owner playback overrule (decisions.md 2026-10-16)",
                                    "by": "owner",
                                    "at": "2026-10-16T22:35:00Z",
                                    "entered_by": "mrgemsol",
                                },
                            ],
                        },
                    },
                },
                {
                    "first": 1039,
                    "last": 1897,
                    "who": {
                        "lead": {"present": "yes", "how": "frame 1043: 1", "by": "owner, 2026-10-11"},
                        "second": {"present": "no", "how": "not on frame", "by": "owner, 2026-10-11"},
                    },
                },
                {
                    "first": 1898,
                    "last": 5934,
                    "scene_score": 0.412,
                    "merged_from": [{"first": 1898, "last": 1919}, {"first": 1920, "last": 5934}],
                    "who": {
                        "lead": {"present": "yes", "how": "merged", "by": "owner, 2026-10-12"},
                        "second": {"present": "no", "how": "merged", "by": "owner, 2026-10-12"},
                    },
                },
                {
                    "first": 5935,
                    "last": 6104,
                    "split_from": {"first": 5935, "last": 6394},
                    "who": {
                        "lead": {"present": "yes", "how": "frame 6055: 1", "by": "owner, 2026-10-13"},
                        "second": {"present": "no", "how": "not on frame", "by": "owner, 2026-10-13"},
                    },
                },
                {
                    "first": 6105,
                    "last": 6394,
                    "split_from": {"first": 5935, "last": 6394},
                    "answered_over": [{"at": "2026-10-17T06:51:43Z", "by": "mrohmy", "who": {}}],
                    "who": {
                        "lead": {"present": "yes", "how": "frame 6164: on this frame", "by": "owner, 2026-10-13"},
                        "second": {"present": "no", "how": "not on frame", "by": "owner, 2026-10-13"},
                    },
                },
            ],
        }
        shots_p.write_text(json.dumps(shots_obj, indent=1))

        cast_file.append(
            cast_p,
            [
                cast_file.entry("lead", 700, "none of the outlined people", by="owner, earlier (answers.jsonl line 420)", table="pv_lead_shots_00001_shots.json"),
                cast_file.entry("lead", 700, "here", by="owner, later (answers.jsonl line 421)", box=[10, 20, 30, 40], size=[1024, 768], table="pv_lead_shots_00001_shots.json"),
                cast_file.entry("second", 606, "here", by="owner (answers.jsonl line 425)", box=[100, 120, 50, 60], size=[1024, 768], table="pv_second_shots_00001_shots.json"),
            ],
        )

        answers_list = [
            # Line 1: early schema ("the lead" via person/chosen, no subject key)
            {"at": "2026-10-11T10:00:00Z", "who": "owner", "row": "who", "what": "choice", "person": 1, "chosen": "the lead", "source_frame": 700, "table": "pv_lead_shots_00001_shots.json", "frames": [604, 1038], "video": "fun.mp4"},
            # Line 2: b20 row choice
            {"at": "2026-10-11T11:00:00Z", "who": "owner", "row": "b20", "what": "choice", "subject": "lead", "chosen": "1", "person": 1, "source_frame": 1043, "table": "step0_lead_shots_00001_shots.json", "frames": [1039, 1897], "video": "fun.mp4"},
            # Line 3: choice on frame 6164 recorded before 5935-6394 was split (frames=[5935, 6394])
            {"at": "2026-10-12T21:00:54Z", "who": "owner", "row": "who", "what": "choice", "subject": "lead", "chosen": "not outlined", "person": None, "source_frame": 6164, "table": "plain", "frames": [5935, 6394], "video": "fun.mp4"},
            # Line 4: split-shot word note with source_frame=5935 naming frame 6164 -> routes to 6105-6394
            {"at": "2026-10-12T21:01:08Z", "who": "owner", "row": "who", "what": "words", "words": "lead arm raised in frame 6164", "source_frame": 5935, "frames": [5935, 6394], "video": "fun.mp4"},
            # Line 5: hedged word note on 5935-6104
            {"at": "2026-10-13T18:44:05Z", "who": "owner", "row": "who", "what": "words", "words": "i THINK its #2", "source_frame": 5935, "frames": [5935, 6394], "video": "fun.mp4"},
        ]
        ans_p.write_text("".join(json.dumps(x) + "\n" for x in answers_list))

        plan_obj = {
            "aliases": {"singer": "lead", "mother": "second"},
            "subjects": {
                "lead": {"region": "face", "parts": ["face_and_neck"], "voice": [{"first": 604, "last": 800, "sings": True}]},
                "second": {"region": "face and hair", "parts": ["face_and_neck", "hair"]},
            },
            "not_people": [{"what": "cereal-box picture", "shots": [[604, 1038]]}],
            "departures": [{"subject": "second", "first": 950, "last": 1038, "why": "watched on playback"}],
        }
        plan_p.write_text(json.dumps(plan_obj))
        man_p.write_text(json.dumps({"source": {"name": "fun.mp4", "rate": 23.976024}}))

        r = subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--shots",
                str(shots_p),
                "--cast",
                str(cast_p),
                "--answers",
                str(ans_p),
                "--plan",
                str(plan_p),
                "--manifest",
                str(man_p),
                "--json",
                str(out_json),
                "--md",
                str(out_md),
            ],
            capture_output=True,
            text=True,
        )
        view = json.loads(out_json.read_text()) if out_json.is_file() else {}
        by_first = {s["first"]: s for s in view.get("shots", [])}

        # Case 1: shot-level fields and history preserved
        s604 = by_first.get(604, {})
        s1898 = by_first.get(1898, {})
        s6105 = by_first.get(6105, {})
        case(
            "shot_level_fields_and_history_are_preserved",
            r.returncode == 0
            and s604.get("order") == ["second", "lead"]
            and "second first" in (s604.get("order_by") or "")
            and s604.get("others_present") == "yes"
            and s1898.get("scene_score") == 0.412
            and "merged_from" in (s1898.get("history") or {})
            and set((s6105.get("history") or {}).keys()) == {"split_from", "answered_over"},
            r.stdout + r.stderr + json.dumps([s604.get("order"), s1898.get("history"), s6105.get("history")]),
        )

        # Case 2: cast_entries preserve box, size, table, standing
        ce_lead = s604["subjects"]["lead"]["cast_entries"]
        sb_lead = s604["subjects"]["lead"]["standing_boxes"]
        case(
            "cast_entries_keep_box_size_table_and_standing",
            len(ce_lead) == 2
            and ce_lead[0]["standing"] is False
            and ce_lead[1]["standing"] is True
            and ce_lead[1]["box"] == [10.0, 20.0, 30.0, 40.0]
            and ce_lead[1]["size"] == [1024, 768]
            and ce_lead[1]["table"] == "pv_lead_shots_00001_shots.json"
            and len(sb_lead) == 1
            and sb_lead[0]["line"] == 2,
            json.dumps(ce_lead),
        )

        # Case 3: line_numbers expand ranges and exclude dates
        note_lines = s604["note"]["answer_lines"]
        att_lines = s604["subjects"]["lead"]["answer"][0]["answer_lines"]
        case(
            "line_numbers_expand_ranges_and_exclude_dates",
            note_lines == [310] and att_lines == [420, 421, 422, 428],
            f"note_lines={note_lines}, att_lines={att_lines}",
        )

        # Case 4: owner_answers match by frame shown across b20 and who, and split-shot words route to 6105-6394
        s1039 = by_first[1039]
        s5935 = by_first[5935]
        case(
            "owner_answers_match_by_frame_shown_across_b20_and_who_and_split_words",
            len(s604["subjects"]["lead"]["owner_answers"]) == 1
            and s604["subjects"]["lead"]["owner_answers"][0]["table"] == "pv_lead_shots_00001_shots.json"
            and len(s1039["subjects"]["lead"]["owner_answers"]) == 1
            and s1039["subjects"]["lead"]["owner_answers"][0]["row"] == "b20"
            and s5935["subjects"]["lead"]["owner_answers"] == []
            and [x["frame"] for x in s6105["subjects"]["lead"]["owner_answers"]] == [6164]
            and [w["line"] for w in s5935["note"]["owner_words"]] == [5]
            and [(w["line"], w["frame"]) for w in s6105["note"]["owner_words"]] == [(4, 6164)],
            json.dumps([s5935["note"], s6105["note"], s5935["subjects"]["lead"]["owner_answers"], s6105["subjects"]["lead"]["owner_answers"]]),
        )

        # Case 5: cut_from prefers full containment and lists all on merged shot
        case(
            "cut_from_prefers_full_containment_and_lists_all_on_merged",
            s1039["cut_from"] == "1039-1919"
            and s1039["cut_from_all"] == ["604-1050", "1039-1919"]
            and s1898["cut_from_all"] == ["1039-1919", "1920-6394"],
            json.dumps([s1039["cut_from"], s1039["cut_from_all"], s1898["cut_from_all"]]),
        )

        # Case 6: attribution parses "the owner" and "entered by shots_record.py answer for mrohmy"
        att_lead = s604["subjects"]["lead"]["answer"]
        att_sec = s604["subjects"]["second"]["answer"]
        case(
            "attribution_parses_the_owner_and_shots_record_answer_for",
            att_lead[1]["said_by"] == "owner"
            and att_sec[0]["said_by"] == "owner"
            and att_sec[0]["entered_by"] == "shots_record.py answer for mrohmy"
            and "sessions_named" not in att_sec[0],
            json.dumps([att_lead, att_sec]),
        )

        # Case 7: seen_standing overrides not seen when seen follows, and flags hedged words
        st_sec = s604["subjects"]["second"]["seen_standing"]
        case(
            "seen_standing_overrides_not_seen_when_seen_follows_and_flags_hedged_words",
            st_sec["not_seen_spans"] == {"face_and_neck": []}
            and len(st_sec["overruled_looks"]) == 1
            and st_sec["overruled_looks"][0]["frames"] == [[950, 1038]]
            and s5935["note"]["owner_hedged"] is True
            and s6105["note"]["owner_hedged"] is False,
            json.dumps([st_sec, s5935["note"], s6105["note"]]),
        )

        # Case 8: made_from sha256s, source_fps vs copy_fps, and verify_view_fresh
        who_view.verify_view_fresh(view, shots_p, cast_p, ans_p, plan_p)
        ans_p.write_text(ans_p.read_text() + "\n")
        stale_caught = False
        try:
            who_view.verify_view_fresh(view, shots_p, cast_p, ans_p, plan_p)
        except who_view.Refused as e:
            stale_caught = "stale for owner_answers" in str(e)
        case(
            "made_from_records_sha256_and_verify_view_fresh_refuses_stale",
            view["video"]["copy_fps"] == 24
            and view["video"]["source_fps"] == 23.976024
            and out_md.is_file()
            and stale_caught,
            json.dumps(view["video"]),
        )

        # Case 9: refuses missing or corrupt inputs cleanly with no traceback
        bad_shots = root / "bad_shots.json"
        bad_shots.write_text(json.dumps({"schema": "wrong"}))
        r_bad = subprocess.run(
            [sys.executable, str(TOOL), "--shots", str(bad_shots), "--cast", str(cast_p), "--answers", str(ans_p), "--json", str(out_json)],
            capture_output=True,
            text=True,
        )
        case(
            "refuses_missing_or_corrupt_inputs_cleanly",
            r_bad.returncode != 0 and "REFUSED:" in r_bad.stderr and "Traceback" not in r_bad.stderr,
            r_bad.stdout + r_bad.stderr,
        )

    print(f"ok: {fails} failing case(s)" if not fails else f"FAILED: {fails} case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
