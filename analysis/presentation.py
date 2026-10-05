"""Write a single offline page containing the verification results."""

from html import escape
import os

from .common import ROOT, read, rows


def table(headers, records):
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in records
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build(output, result):
    def csv(name):
        return rows(output / "tables" / f"{name}.csv")

    def number(row, key, digits=2):
        return f"{float(row[key]):.{digits}f}"

    def case(name):
        size, view = name.split("_")
        return f"{size[1:]} × {size[1:]}, {view}"

    def policy(name):
        if name.startswith("dqn"):
            return "DQN"
        return "BFS" if name == "bfs" else name.capitalize()

    def download(name):
        return f"<a href='tables/{name}.csv'>CSV</a>"

    report = escape(
        os.path.relpath(ROOT / "report/snake_rl_report.pdf", output).replace("\\", "/")
    )
    sections = []
    sections.append(
        "<h2 id='native'>Table 1 · Original Snake</h2>" + download("table1_native")
    )
    sections.append(
        table(
            ["Board / view", "Policy", "Fruits", "Wall hits", "Body hits", "Return"],
            [
                [
                    case(r["case"]),
                    policy(r["policy"]),
                    *[
                        number(r, key + "_mean")
                        for key in ("fruits", "wall_hits", "body_hits", "return")
                    ],
                ]
                for r in csv("table1_native")
            ],
        )
    )
    native = {
        (r["case"], r["policy"]): r
        for r in csv("native_all_policies")
        if r["policy"] in ("greedy", "bfs", "direct")
    }
    sections.append("<h3>Heuristic returns</h3>" + download("native_all_policies"))
    sections.append(
        table(
            ["Board / view", "Greedy", "BFS", "Direct"],
            [
                [
                    case(name),
                    *[
                        number(native[name, p], "return_mean")
                        for p in ("greedy", "bfs", "direct")
                    ],
                ]
                for name in ("b7_full", "b7_partial", "b11_full", "b11_partial")
            ],
        )
    )
    sections.append(
        "<h3>Paired DQN − Direct return differences</h3>"
        + download("native_paired_returns")
    )
    sections.append(
        table(
            ["Board / view", "Difference", "95% CI"],
            [
                [
                    case(r["case"]),
                    number(r, "difference"),
                    f"[{number(r, 'ci95_low')}, {number(r, 'ci95_high')}]",
                ]
                for r in csv("native_paired_returns")
            ],
        )
    )
    sections.append(
        "<p>Episode IDs are paired. Intervals use the sample standard deviation of the differences, with 1.96 standard errors. They describe frozen policies, not variability across training seeds.</p>"
    )

    sections.append(
        "<h2 id='search'>Table 2 · Hidden-fruit discovery</h2>"
        + download("table2_search")
    )
    search = {(r["case"], policy(r["policy"])): r for r in csv("table2_search")}
    sections.append(
        table(
            ["Board / view", "Hidden starts", "Direct wait", "DQN wait"],
            [
                [
                    case(name),
                    search[name, "Direct"]["hidden_starts"],
                    number(search[name, "Direct"], "initial_wait_mean"),
                    number(search[name, "DQN"], "initial_wait_mean"),
                ]
                for name in ("b7_partial", "b11_partial")
            ],
        )
    )
    sections.append(
        "<p>Wait is measured on identical initially hidden starts; failures count as 1,000 actions.</p>"
    )

    sections.append(
        "<h2 id='classic'>Table 3 · Classic Snake</h2>" + download("table3_classic")
    )
    sections.append(
        table(
            [
                "View",
                "Policy",
                "Wins / 500",
                "Fruits",
                "Deaths",
                "Timeouts",
                "Completion: 95% Wilson CI",
            ],
            [
                [
                    r["regime"],
                    policy(r["policy"]),
                    r["wins"],
                    number(r, "fruits_mean"),
                    r["deaths"],
                    r["timeouts"],
                    f"{100 * float(r['completion_rate']):.1f}% [{100 * float(r['completion_ci95_low']):.2f}, {100 * float(r['completion_ci95_high']):.2f}]%",
                ]
                for r in csv("table3_classic")
            ],
        )
    )
    sections.append(
        "<p>All 47 full and 342 partial DQN timeouts are exact physical-state cycles. Detection does not alter play.</p>"
    )

    labels = {
        "full_control": "Full · control",
        "full_lower_lr": "Full · lower LR",
        "full_watkins": "Full · Watkins-style",
        "full_discount": "Full · γ = 0.995",
        "extended_partial": "Partial · control",
        "partial_protected": "Partial · protected replay",
    }
    sections.append(
        "<h2 id='interventions'>Table 4 · Classic continuations</h2>"
        + download("table4_interventions")
    )
    sections.append(
        "<p>Each window adds 1.024M transitions. Initial full score: 55 wins / 16.99 fruits at 4.096M; initial partial: 3 / 12.20 at 2.432M. Best new excludes that starting checkpoint.</p>"
    )
    sections.append(
        table(
            [
                "Continuation",
                "Best new: wins / fruits",
                "Checkpoint (M)",
                "Last four: wins / fruits",
                "Final wins",
            ],
            [
                [
                    labels[r["run"]],
                    f"{r['best_new_wins']} / {number(r, 'best_new_fruits')}",
                    f"{int(r['best_new_counter']) / 1e6:.3f}",
                    f"{float(r['last_four_wins']):g} / {number(r, 'last_four_fruits')}",
                    r["final_wins"],
                ]
                for r in csv("table4_interventions")
            ],
        )
    )

    sections.append(
        "<h2 id='figures'>Learning curves</h2><p>Rebuilt from the archived validation histories. Without smoothing; shared native prefixes and the full Classic control segment are retained.</p>"
    )
    for name, title in (
        ("native_learning", "Figure 1 · Original Snake"),
        ("classic_learning", "Figure 2 · Classic Snake"),
    ):
        sections.append(
            f"<figure><figcaption>{title} · <a href='figures/{name}.pdf'>vector PDF</a></figcaption><img src='figures/{name}.png' alt='{title}'></figure>"
        )

    sections.append(
        "<details><summary>Checkpoint selection and supporting diagnostics</summary>"
    )
    choices = read(output / "selection.json")
    selections = []
    for name, item in choices.items():
        criterion = (
            "last-eight validation return, then earliest maximum checkpoint"
            if name.startswith("b")
            else "validation wins, then fruits, then earlier tie"
        )
        selections.append(
            [
                name,
                f"{item['selected_transitions'] / 1e6:.3f}M",
                item.get("winner", "—"),
                criterion,
            ]
        )
    sections.append(
        table(["Agent", "Selected transitions", "Schedule", "Criterion"], selections)
    )
    sections.append(
        "<h3>Learning-rate schedule selection</h3><a href='selection.json'>JSON</a>"
    )
    sections.append(
        table(
            ["Board / view", "Schedule", "Last-eight validation return", "Chosen"],
            [
                [
                    case(name),
                    schedule["variant"],
                    f"{schedule['final_eight_mean']:.2f}",
                    "Yes" if schedule["variant"] == item["winner"] else "",
                ]
                for name, item in choices.items()
                if "schedules" in item
                for schedule in item["schedules"]
            ],
        )
    )
    sections.append(
        "<p>Choose the highest mean of the final eight validation evaluations; within that schedule, retain the earliest maximum-return checkpoint. Test scores do not determine selection.</p>"
    )
    sections.append("<h3>Expected spawn distance</h3>" + download("spawn_geometry"))
    sections.append(
        table(
            ["Board / view", "Policy", "Expected distance", "Body effect"],
            [
                [
                    case(r["case"]),
                    policy(r["policy"]),
                    number(r, "expected_distance", 4),
                    number(r, "body_effect", 4),
                ]
                for r in csv("spawn_geometry")
            ],
        )
    )
    sections.append(
        "<h3>Search failures and pursuit</h3>" + download("table2_search_all")
    )
    sections.append(
        table(
            ["Board / view", "Policy", "Zero fruit, never seen", "Pursuit excess"],
            [
                [
                    case(r["case"]),
                    r["policy"],
                    r["zero_fruit_unseen"],
                    number(r, "pursuit_excess", 3),
                ]
                for r in csv("table2_search_all")
            ],
        )
    )
    replay = read(ROOT / "evidence/diagnostics/replay.json")
    sections.append(
        "<h3>Final replay events</h3><p>Counts are recomputed from all 100,000 delivered slots; state and optimizer tensors are unnecessary for these counts.</p>"
    )
    sections.append(
        table(
            [
                "View",
                "Slots",
                "Winning terminals",
                "Death terminals",
                "Truncations",
                "Fruit steps",
            ],
            [
                [
                    r["regime"],
                    r["replay_size"],
                    r["terminal_wins"],
                    r["terminal_deaths"],
                    r["truncations"],
                    r["fruit_steps"],
                ]
                for r in replay["snapshots"]
            ],
        )
    )
    sections.append("</details>")

    live = result.get("live_checks")
    if result["mode"] == "full":
        scope = "Independent replay passed: all 11,000 final-policy episodes and 4,000 native diagnostic trajectories match the archived records."
    elif result["mode"] == "final":
        scope = "Final test passed: all 8,000 native and 3,000 Classic episode records match the archive. Both environments use the held-out test split, with 500 episodes per policy and configuration."
    elif live:
        scope = (
            f"Live sample passed: {live['native_games']} native validation episodes and "
            f"{live['classic_records_compared']} Classic held-out episodes. Classic records match "
            "the corresponding archived games exactly. Native's smaller batch has its own RNG "
            "allocation; it is a functional sample, not a prefix of the report bank."
        )
    else:
        scope = "Archive-only run: no episodes were simulated."
    log_link = " · <a href='evaluation.log'>Evaluation log</a>" if live else ""
    header = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Snake RL · Verification results</title><style>
body{{max-width:1080px;margin:40px auto;padding:0 24px;color:#202833;background:#fff;font:16px/1.55 Arial,sans-serif}}
h1,h2,h3{{line-height:1.25}}h2{{margin-top:44px;border-bottom:1px solid #dae1e8;padding-bottom:10px}}
a{{color:#175c93}}nav a{{display:inline-block;margin:0 16px 8px 0}}.status{{border-left:4px solid #287a4d;background:#edf6f0;padding:16px 20px}}
.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:12px 0 20px}}th,td{{padding:8px 10px;border-bottom:1px solid #e1e5eb;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}thead{{background:#f0f3f6}}figure{{margin:24px 0}}img{{width:100%;height:auto}}figcaption{{font-weight:bold}}summary{{cursor:pointer;font-weight:bold}}details{{margin-top:32px}}
</style></head><body><h1>Snake RL · Verification results</h1>
<p><a href='{report}'>Submitted report PDF</a> · <a href='verification.json'>Machine-readable checks</a>{log_link}</p>
<div class='status'><strong>Passed</strong> · {result['report']['printed_values_checked']} printed report values matched · 55 evidence files checked · seed 0<br>
{escape(scope)}<br>Elapsed: {result['seconds']:.1f} s. No training was performed.</div>
<p>The tables and figures below are reconstructed from the report's archived data: 500 games per test row, 100 validation games per learning point. Checkpoint selection uses validation only. The optional --quick sample is separate from the final test bank.</p>
<nav><a href='#native'>Table 1</a><a href='#search'>Table 2</a><a href='#classic'>Table 3</a><a href='#interventions'>Table 4</a><a href='#figures'>Figures</a></nav>
"""
    (output / "index.html").write_text(
        header + "\n".join(sections) + "</body></html>\n", encoding="utf-8"
    )
