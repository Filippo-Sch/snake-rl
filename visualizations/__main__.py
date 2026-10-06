"""Build the interactive replay and optional LinkedIn video."""

import argparse
import json
from pathlib import Path

from .replay import ROOT, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="store_true",
        help="Also export the portrait MP4; requires FFmpeg",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="FFmpeg executable or absolute path"
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse the previously verified replay cache",
    )
    args = parser.parse_args()
    output = ROOT / "results/visualizations"
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "replays.json"
    if args.reuse:
        data = json.loads(cache.read_text(encoding="utf-8"))
        import hashlib
        from .replay import read

        manifest = read(ROOT / "weights/manifest.json")
        for name, value in data["checkpoints"].items():
            item = manifest["policies"][name]
            assert (
                hashlib.sha256(
                    (ROOT / "weights" / item["file"]).read_bytes()
                ).hexdigest()
                == value
            )
    else:
        data = record()
        cache.write_text(
            json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    template = (ROOT / "visualizations/page.html").read_text(encoding="utf-8")
    html = template.replace("__REPLAY_DATA__", json.dumps(data, separators=(",", ":")))
    (ROOT / "visualizations/index.html").write_text(
        html, encoding="utf-8", newline="\n"
    )
    verification = dict(
        status="passed",
        seed=0,
        training=False,
        archived_records_matched=data["matched_records"],
        episodes=[
            dict(
                id=e["id"],
                episode_id=e["episode_id"],
                actions=e["frames"][-1]["t"],
                fruits=e["frames"][-1]["fruits"],
                outcome=e["frames"][-1]["outcome"],
            )
            for e in data["episodes"]
        ],
    )
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    from .render import poster, video

    poster(data, ROOT / "visualizations/poster.png")
    if args.video:
        video(data, output / "snake_rl_linkedin.mp4", args.ffmpeg)
    print("Interactive replay:", ROOT / "visualizations/index.html")
    print(
        "Verified against the original held-out evidence:",
        data["matched_records"],
        "records.",
    )


if __name__ == "__main__":
    main()
