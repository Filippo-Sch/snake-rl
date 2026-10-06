"""Regenerate the single self-contained interactive replay page."""

import argparse
import hashlib
import json
import re

from .replay import ROOT, read, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse the verified version-2 replay cache"
    )
    args = parser.parse_args()
    output = ROOT / "results/visualizations"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "replays.json"
    if args.reuse:
        data = read(cache)
        if data.get("format") != "snake-visual-replay-v2":
            parser.error(
                "The replay cache is older than this viewer; regenerate without --reuse"
            )
        manifest = read(ROOT / "weights/manifest.json")
        for name, expected in data["checkpoints"].items():
            path = ROOT / "weights" / manifest["policies"][name]["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checkpoint changed: {name}")
    else:
        data = record()
        cache.write_text(
            json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    page = ROOT / "index.html"
    html = page.read_text(encoding="utf-8")
    pattern = r'(<script id="replay-data" type="application/json">).*?(</script>)'
    payload = json.dumps(data, separators=(",", ":"))
    html, replacements = re.subn(
        pattern, lambda match: match[1] + payload + match[2], html, count=1, flags=re.S
    )
    if replacements != 1:
        raise ValueError("index.html must contain one replay-data script element")
    page.write_text(html, encoding="utf-8", newline="\n")
    verification = dict(
        status="passed",
        seed=0,
        training=False,
        archived_records_matched=data["matched_records"],
        partial_success_id=data["partial_success_id"],
        baseline_ranking=data["baseline_ranking"],
        episodes=[
            dict(
                id=e["id"],
                policy=e["policy"],
                episode_id=e["episode_id"],
                actions=e["frames"][-1]["t"],
                fruits=e["frames"][-1]["fruits"],
                outcome=e["frames"][-1]["outcome"],
                legal_tail_entries=sum(bool(f.get("tail_entry")) for f in e["frames"]),
            )
            for e in data["episodes"]
        ],
    )
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print("Interactive replay:", page)
    print("Matched archived records:", data["matched_records"])


if __name__ == "__main__":
    main()
