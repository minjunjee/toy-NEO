#!/usr/bin/env python3
"""Build docs/insights-by-experiment.md from scripts/insight_history/.

Every experiment of the repository in order. For each: what was asked and done, how its setting differs from the lab
NEO (a table of the parts that differ, and a line under every figure and table), its figures and tables (tables are
copied from the documents that hold them, so the numbers are not retyped), and the state of everything known about NEO
at that point: what the experiment added, what it corrected (with the earlier statement), what it discarded, and the
rest unchanged. The last section is the current state in full.

An insight is an entry with a short and a full statement. Experiments add entries, revise them or retire them
(scripts/insight_history/part*.py); the state under each experiment is replayed from those operations, so the summaries
cannot disagree with each other.

    python scripts/build_insight_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from insight_history import part1, part2, part3  # noqa: E402
from insight_history.base import LAB, TOPICS  # noqa: E402

DOCS = REPO / "docs"


def find_table(source: str, header: str, nth: int = 0) -> list[str]:
    """The nth table of docs/<source> whose first row contains `header`."""
    lines = (DOCS / source).read_text().split("\n")
    seen = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|")) and header in line:
            if seen == nth:
                block = []
                while i < len(lines) and lines[i].startswith("|"):
                    block.append(lines[i])
                    i += 1
                return block
            seen += 1
    raise KeyError(f"no table with {header!r} (occurrence {nth}) in {source}")


def diff_table(diff: dict, column: str = "이 실험") -> list[str]:
    out = [f"| 항목 | 랩 NEO | {column} |", "|---|---|---|"]
    for field, lab in LAB.items():
        if field in diff:
            out.append(f"| {field} | {lab} | {diff[field]} |")
    return out


def main() -> None:
    experiments = part1.EXPERIMENTS + part2.EXPERIMENTS + part3.EXPERIMENTS
    out = ["# 실험별 figure와 table, 랩 NEO와 다른 점, 그리고 그 시점까지의 insight", ""]
    out += [part1.INTRO, ""]
    out += ["## 랩 NEO의 기준 설정", "", part1.LAB_NOTE, "", "| 항목 | 랩 NEO |", "|---|---|"]
    out += [f"| {field} | {value} |" for field, value in LAB.items()]
    out.append("")
    live: dict[str, dict] = {}          # id -> {topic, short, full, since}
    order: list[str] = []
    for n, exp in enumerate(experiments, start=1):
        out += [f"## {n}. {exp['title']}", "", f"{exp['date']}. 근거는 {exp['source']}.", ""]
        out += [f"**물은 것.** {exp['question']}", "", f"**한 것.** {exp['did']}", ""]
        out += ["**랩 NEO와 다른 점, 이 실험의 공통 설정.**" + (" " + exp["diff_note"] if exp.get("diff_note") else ""), ""]
        if exp["diff"]:
            out += diff_table(exp["diff"]) + ["", "표에 없는 항목은 랩 NEO와 같다.", ""]
        figure = table = 0
        for item in exp["items"]:
            kind = item[0]
            if kind == "fig":
                _, path, caption, note = item
                assert (DOCS / path).exists(), path
                figure += 1
                out += [f"**Figure {n}.{figure}.** {caption}", "", f"![{caption}]({path})", "", f"랩 NEO와 다른 점: {note}", ""]
            else:
                table += 1
                if kind == "table":
                    _, source, header, caption, note, *rest = item
                    block = find_table(source, header, rest[0] if rest else 0)
                else:
                    _, block, caption, note = item
                    block = block.strip().split("\n")
                out += [f"**Table {n}.{table}.** {caption}", ""] + block + ["", f"랩 NEO와 다른 점: {note}", ""]
        out += [f"**나온 것.** {exp['result']}", ""]
        # the state of the insights after this experiment
        new, revised, retired = [], [], []
        for op in exp["insights"]:
            if op[0] == "new":
                _, key, topic, short, full = op
                assert key not in live, key
                live[key] = {"topic": topic, "short": short, "full": full}
                order.append(key)
                new.append(key)
            elif op[0] == "revise":
                _, key, short, full, why = op
                revised.append((key, live[key]["full"], why))
                live[key].update(short=short, full=full)
            else:
                _, key, why = op
                retired.append((live[key]["full"], why))
                del live[key]
                order.remove(key)
        touched = set(new) | {key for key, _, _ in revised}
        out += [f"### {n}번째 실험까지의 insight", ""]
        if new:
            out += ["새로 알게 된 것.", ""] + [f"- {live[key]['full']}" for key in new] + [""]
        if revised:
            out += ["고쳐진 것.", ""]
            for key, before, why in revised:
                out += [f"- 앞서 한 말: {before}", f"  지금: {live[key]['full']}", f"  무엇을 보고: {why}"]
            out.append("")
        if retired:
            out += ["버린 것.", ""] + [f"- {before} 버린 이유: {why}" for before, why in retired] + [""]
        rest = [key for key in order if key not in touched]
        if rest:
            out += ["그대로인 것.", ""]
            for topic, name in TOPICS:
                keys = [key for key in rest if live[key]["topic"] == topic]
                if keys:
                    out.append(f"- {name}")
                    out += [f"    - {live[key]['short']}" for key in keys]
            out.append("")
    out += ["## 지금의 정리", "", part3.CLOSING, ""]
    for topic, name in TOPICS:
        keys = [key for key in order if live[key]["topic"] == topic]
        if keys:
            out += [f"### {name}", ""] + [f"- {live[key]['full']}" for key in keys] + [""]
    out += ["### 랩 NEO에서 지금의 최선까지 달라진 것", "", part3.BEST_NOTE, ""] + diff_table(part3.BEST, "지금의 최선") + [""]
    target = DOCS / "insights-by-experiment.md"
    target.write_text("\n".join(out) + "\n")
    print(f"{target}: {len(experiments)} experiments, {len(order)} insights standing, {target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
