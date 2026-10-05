"""Compare recomputed results with the submitted report's printed values."""

import re

from .common import ROOT, SELECTED, contained, read, require, sha


def table_rows(source, label):
    blocks = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source, re.S)
    block = next(b for b in blocks if f"\\label{{{label}}}" in b)
    return [
        line.strip().removesuffix(r"\\").strip().split("&")
        for line in block.splitlines()
        if "&" in line and line.rstrip().endswith(r"\\")
    ]


def decimal_values(text):
    return [float(n) for n in re.findall(r"[-+]?\d+\.\d+", text)]


def check_report(tables, choices):
    manifest = read(ROOT / "report/manifest.json")
    for name, digest in manifest["files"].items():
        require(
            sha(contained(ROOT / "report", name)) == digest,
            f"Report checksum mismatch: {name}",
        )
    source = (ROOT / "report/source/main.tex").read_text(encoding="utf-8")
    checks = []

    def printed(actual, expected, label, digits=2):
        require(
            f"{float(actual):.{digits}f}" == f"{float(expected):.{digits}f}",
            f"Report value differs: {label}: {actual} versus {expected}",
        )
        checks.append(label)

    native = {(r["case"], r["policy"]): r for r in tables["table1_native"]}
    case = None
    rows = table_rows(source, "tab:native")
    for cells in rows:
        match = re.search(r"\$(\d+)\\times\d+\$ (full|partial)", cells[0])
        if match:
            case = f"b{match[1]}_{match[2]}"
        policy = cells[1].strip().lower()
        if len(cells) == 6 and policy in ("direct", "dqn"):
            record = native[case, SELECTED[case] if policy == "dqn" else policy]
            for field, text in zip(
                ("fruits", "wall_hits", "body_hits", "return"), cells[2:]
            ):
                printed(
                    record[field + "_mean"],
                    float(text),
                    f"Table 1/{case}/{policy}/{field}",
                )
    numeric_rows = [c for c in rows if "multicolumn" in c[0] and "times" in c[0]]
    require(len(numeric_rows) == 4, "Report native comparison rows are incomplete")
    for size, cells in zip((7, 11), numeric_rows[:2]):
        for regime, cell in zip(("full", "partial"), cells[1:]):
            case = f"b{size}_{regime}"
            for policy, value in zip(("greedy", "bfs", "direct"), decimal_values(cell)):
                printed(
                    native[case, policy]["return_mean"],
                    value,
                    f"Table 1/{case}/{policy}/return",
                )
    paired = {r["case"]: r for r in tables["native_paired_returns"]}
    for size, cells in zip((7, 11), numeric_rows[2:]):
        for regime, cell in zip(("full", "partial"), cells[1:]):
            case = f"b{size}_{regime}"
            values = decimal_values(cell)
            require(len(values) == 3, "Report paired interval is incomplete")
            for field, value in zip(("difference", "ci95_low", "ci95_high"), values):
                printed(paired[case][field], value, f"Table 1/{case}/{field}")

    search = {(r["case"], r["policy"]): r for r in tables["table2_search"]}
    for cells in table_rows(source, "tab:search"):
        match = re.search(r"\$(\d+)\\times\d+\$", cells[0])
        if not match:
            continue
        case = f"b{match[1]}_partial"
        printed(
            search[case, "direct"]["hidden_starts"],
            float(cells[1]),
            f"Table 2/{case}/starts",
            0,
        )
        for policy, cell in zip(("direct", SELECTED[case]), cells[2:]):
            printed(
                search[case, policy]["initial_wait_mean"],
                float(cell),
                f"Table 2/{case}/{policy}/wait",
            )

    classic = {(r["regime"], r["policy"]): r for r in tables["table3_classic"]}
    regime = None
    for cells in table_rows(source, "tab:classic"):
        if cells[0].strip().lower() in ("full", "partial"):
            regime = cells[0].strip().lower()
        policy = cells[1].strip().lower()
        if policy not in ("bfs", "greedy", "dqn"):
            continue
        for field, cell in zip(
            ("wins", "fruits_mean", "deaths", "timeouts"), cells[2:]
        ):
            printed(
                classic[regime, policy][field],
                float(cell),
                f"Table 3/{regime}/{policy}/{field}",
                2 if field == "fruits_mean" else 0,
            )

    interventions = {r["run"]: r for r in tables["table4_interventions"]}
    runs = iter(
        (
            "full_control",
            "full_lower_lr",
            "full_watkins",
            "full_discount",
            "extended_partial",
            "partial_protected",
        )
    )
    for cells in table_rows(source, "tab:interventions"):
        if "/" not in cells[2]:
            continue
        run = next(runs)
        values = [float(n.strip()) for cell in cells[2:] for n in cell.split("/")]
        for field, value in zip(
            ("best_new_wins", "best_new_fruits", "last_four_wins", "last_four_fruits"),
            values,
        ):
            printed(interventions[run][field], value, f"Table 4/{run}/{field}")
    require(
        len([c for c in table_rows(source, "tab:interventions") if "/" in c[2]]) == 6,
        "Report intervention rows are incomplete",
    )

    schedules = re.search(
        r"means \(A/B/C\) are ([\d./]+) in full and ([\d./]+) in partial", source
    )
    require(schedules is not None, "Report LR schedule means missing")
    for regime, group in zip(("full", "partial"), schedules.groups()):
        for record, value in zip(
            choices[f"b11_{regime}"]["schedules"], group.split("/")
        ):
            printed(
                record["final_eight_mean"],
                value,
                f"Schedule/{regime}/{record['variant']}",
            )
    geometry = re.search(r"gives ([\d.]+) for Direct and ([\d.]+) for DQN", source)
    require(geometry is not None, "Report spawn distances missing")
    for policy, value in zip(("direct", SELECTED["b7_full"]), geometry.groups()):
        row = next(
            r
            for r in tables["spawn_geometry"]
            if r["case"] == "b7_full" and r["policy"] == policy
        )
        printed(row["expected_distance"], value, f"Spawn/{policy}", 4)
    for regime, digits in (("full", 1), ("partial", 2)):
        row = classic[regime, "dqn"]
        interval = f"{100 * row['completion_ci95_low']:.{digits}f}--{100 * row['completion_ci95_high']:.{digits}f}"
        require(interval in source, f"Report Wilson interval differs: {regime}")
        checks.append(f"Wilson/{regime}")
    require(len(checks) == 120, f"Incomplete report comparison: {len(checks)} values")
    earlier = next(
        r
        for r in tables["table2_search_all"]
        if r["case"] == "b11_partial" and r["policy"] == "dqn_2048000"
    )
    require(
        earlier["zero_fruit_unseen"] == 247 and earlier["zero_fruit_seen"] == 0,
        "Earlier partial search failures differ",
    )
    final_search = search["b11_partial", SELECTED["b11_partial"]]
    require(
        final_search["zero_fruit_unseen"] + final_search["zero_fruit_seen"] == 0,
        "Selected partial policy has zero-fruit games",
    )
    require(
        f"{final_search['pursuit_excess']:.2f}" == "0.68",
        "Partial pursuit excess differs",
    )
    partial = read(ROOT / "evidence/classic/runs/extended_partial/run.json")["history"]
    selected_partial = choices["classic_partial"]
    require(
        selected_partial["wins"] == 3
        and f"{selected_partial['fruits']:.2f}" == "12.20",
        "Classic partial selection score differs",
    )
    require(
        partial[-1]["summary"]["wins"] == 0
        and f"{partial[-1]['summary']['mean_fruits']:.2f}" == "14.16",
        "Classic partial endpoint differs",
    )
    highest_return = max(partial, key=lambda h: h["summary"]["mean_reward"])
    require(
        highest_return["summary"]["wins"] == 1, "Classic return-based selection differs"
    )
    control = read(ROOT / "evidence/classic/runs/full_control/run.json")["history"]
    require(
        all(h["summary"]["wins"] == 0 for h in control[-5:]),
        "Full control's last five evaluations differ",
    )
    require(
        all(r["final_wins"] == 0 for r in tables["table4_interventions"]),
        "An intervention endpoint has wins",
    )
    replay = read(ROOT / "evidence/diagnostics/replay.json")
    require(
        next(s for s in replay["snapshots"] if s["regime"] == "partial")[
            "terminal_wins"
        ]
        == 0,
        "Partial replay contains winning terminals",
    )
    return dict(
        status="passed",
        printed_values_checked=len(checks),
        behavioral_claims_checked=9,
        checks=checks,
        pdf_sha256=manifest["files"]["snake_rl_report.pdf"],
    )
