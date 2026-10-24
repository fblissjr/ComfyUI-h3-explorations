#!/usr/bin/env python3
"""Census of gate flag kinds across every `flags.json` and `outcomes.json` under `--data`.

Usage:
    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/flag_census.py [--data data/]

For every flag kind (`rule`) across every `flags.json` and `outcomes.json` on disk under `--data`, prints as data:
- `flags`: total flag instances written in `flags.json` files
- `loads`: number of capture folders (`flags.json` files) on which the rule fired at least once
- `fail` / `iffy` / `fine`: breakdown by flag level (`likely to fail`, `iffy`, `likely fine`)
- `overridden`: how many times the flag in `flags.json` was overridden (`flags.overridden` or matching `key` in `outcomes.json`)
- `blocked_changed`: how many times the flag blocked a capture/render without being overridden (a `blocked` capture that was
  then changed or superseded, or a non-override outcome in `outcomes.json`)
- `outcome_other`: how many explicit non-override outcomes (`happened: true/false`) exist in `outcomes.json` for this rule
- `last_blocked`: the last date (`YYYY-MM-DD`) the rule blocked without override or recorded a non-override outcome (`-` if never)
- `last_fired`: the last date (`YYYY-MM-DD`) the rule fired in any `flags.json`

Decides nothing; the owner reads the table to decide which flag kinds stay and which go.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class Refused(Exception):
    """A pre-condition that stops the tool with a message and no traceback."""


def flag_key(flag: dict) -> str:
    about = [flag.get(k) for k in ("rule", "subject", "run", "other", "seen_by", "source_frames")]
    return hashlib.sha256(json.dumps(about, sort_keys=True).encode()).hexdigest()[:16]


def _extract_date(written: str | None, folder_name: str) -> str:
    if written and len(written) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", str(written)):
        return str(written)[:10]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", folder_name)
    return m.group(1) if m else ""


def run_census(data_dir: Path) -> dict:
    """Walk every `flags.json` and `outcomes.json` under `data_dir` and return per-rule counts and totals."""
    if not data_dir.is_dir():
        raise Refused(f"{data_dir} is not a folder")
    flags_files = sorted(data_dir.rglob("flags.json"))
    if not flags_files:
        raise Refused(f"no flags.json files found under {data_dir}")

    outcomes_files = sorted(data_dir.rglob("outcomes.json"))
    total_flags_in_files = 0
    by_rule: dict[str, dict] = {}

    def _rule_entry(rule: str) -> dict:
        return by_rule.setdefault(
            rule,
            {
                "rule": rule,
                "flags": 0,
                "loads": 0,
                "likely_to_fail": 0,
                "iffy": 0,
                "likely_fine": 0,
                "overridden": 0,
                "blocked_changed": 0,
                "outcome_other": 0,
                "last_blocked": "",
                "last_fired": "",
            },
        )

    for ff in flags_files:
        cap = ff.parent
        d = json.loads(ff.read_text())
        date = _extract_date(d.get("written"), cap.name)
        overridden_ids = set(d.get("overridden") or [])
        blocking_ids = set(d.get("blocking") or [])
        outcomes_file = cap / "outcomes.json"
        overridden_keys: set[str] = set()
        non_override_keys: set[str] = set()
        non_override_id_rules: set[tuple[str | None, str | None]] = set()
        outcome_dates: dict = {}

        if outcomes_file.is_file():
            od = json.loads(outcomes_file.read_text())
            for o in od.get("outcomes") or []:
                k = o.get("key")
                r = o.get("rule")
                w = _extract_date(o.get("written"), cap.name) or date
                if o.get("overridden"):
                    if k:
                        overridden_keys.add(k)
                else:
                    if r:
                        st_r = _rule_entry(str(r))
                        st_r["outcome_other"] += 1
                        st_r["last_blocked"] = max(st_r["last_blocked"], w)
                    if k:
                        non_override_keys.add(k)
                        outcome_dates[k] = max(outcome_dates.get(k, ""), w)
                    else:
                        pair = (o.get("flag"), r)
                        non_override_id_rules.add(pair)
                        outcome_dates[pair] = max(outcome_dates.get(pair, ""), w)

        seen_rules_in_cap: set[str] = set()
        for fl in d.get("flags") or []:
            total_flags_in_files += 1
            rule = str(fl.get("rule") or "unknown")
            fid = fl.get("id")
            fk = flag_key(fl)
            lvl = fl.get("level")
            st = _rule_entry(rule)
            st["flags"] += 1
            seen_rules_in_cap.add(rule)
            if lvl == "likely to fail":
                st["likely_to_fail"] += 1
            elif lvl == "iffy":
                st["iffy"] += 1
            else:
                st["likely_fine"] += 1
            if date:
                st["last_fired"] = max(st["last_fired"], date)

            is_overridden = (fid in overridden_ids) or (fk in overridden_keys)
            has_other_outcome = (fk in non_override_keys) or ((fid, rule) in non_override_id_rules)
            if is_overridden:
                st["overridden"] += 1
            if ((fid in blocking_ids or lvl == "likely to fail") and not is_overridden) or has_other_outcome:
                st["blocked_changed"] += 1
                chg_date = outcome_dates.get(fk) or outcome_dates.get((fid, rule)) or date
                if chg_date:
                    st["last_blocked"] = max(st["last_blocked"], chg_date)

        for rule in seen_rules_in_cap:
            by_rule[rule]["loads"] += 1

    # Also check any outcomes.json in a folder with no flags.json
    flags_parents = {ff.parent for ff in flags_files}
    for of in outcomes_files:
        if of.parent not in flags_parents:
            od = json.loads(of.read_text())
            for o in od.get("outcomes") or []:
                if not o.get("overridden") and o.get("rule"):
                    st_r = _rule_entry(str(o["rule"]))
                    st_r["outcome_other"] += 1
                    w = _extract_date(o.get("written"), of.parent.name)
                    if w:
                        st_r["last_blocked"] = max(st_r["last_blocked"], w)

    rows = sorted(by_rule.values(), key=lambda r: (-r["flags"], r["rule"]))
    sum_flags = sum(r["flags"] for r in rows)
    return {
        "data_dir": str(data_dir),
        "flags_files_count": len(flags_files),
        "outcomes_files_count": len(outcomes_files),
        "total_flags_in_files": total_flags_in_files,
        "sum_of_rule_flags": sum_flags,
        "rows": rows,
    }


def print_census(result: dict, out=sys.stdout) -> None:
    rows = result["rows"]
    print(
        f"flag census across {result['data_dir']}: {result['flags_files_count']} flags.json file(s), "
        f"{result['outcomes_files_count']} outcomes.json file(s), {result['total_flags_in_files']} total flag(s), "
        f"{len(rows)} flag kind(s)",
        file=out,
    )
    header = (
        f"{'rule':<42s} {'flags':>6s} {'loads':>6s} {'fail':>5s} {'iffy':>6s} {'fine':>5s} "
        f"{'overr':>6s} {'blocked':>7s} {'out_oth':>7s} {'last_blocked':<12s} {'last_fired':<10s}"
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    for r in rows:
        print(
            f"{r['rule']:<42s} {r['flags']:>6d} {r['loads']:>6d} {r['likely_to_fail']:>5d} {r['iffy']:>6d} {r['likely_fine']:>5d} "
            f"{r['overridden']:>6d} {r['blocked_changed']:>7d} {r['outcome_other']:>7d} "
            f"{(r['last_blocked'] or '-'):<12s} {(r['last_fired'] or '-'):<10s}",
            file=out,
        )
    print("-" * len(header), file=out)
    tot_flags = sum(r["flags"] for r in rows)
    tot_fail = sum(r["likely_to_fail"] for r in rows)
    tot_iffy = sum(r["iffy"] for r in rows)
    tot_fine = sum(r["likely_fine"] for r in rows)
    tot_overr = sum(r["overridden"] for r in rows)
    tot_blk = sum(r["blocked_changed"] for r in rows)
    tot_oth = sum(r["outcome_other"] for r in rows)
    print(
        f"{'TOTAL':<42s} {tot_flags:>6d} {result['flags_files_count']:>6d} {tot_fail:>5d} {tot_iffy:>6d} {tot_fine:>5d} "
        f"{tot_overr:>6d} {tot_blk:>7d} {tot_oth:>7d} {'-':<12s} {'-':<10s}",
        file=out,
    )
    print(
        f"\ncheck: sum of rule flags ({tot_flags}) == total flags in files ({result['total_flags_in_files']}); "
        f"fail ({tot_fail}) == overridden ({tot_overr}) + blocked_changed ({tot_blk})",
        file=out,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=REPO / "data", help="folder holding capture folders with flags.json and outcomes.json (default: data/)")
    a = p.parse_args(argv)
    try:
        res = run_census(a.data)
        print_census(res)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
