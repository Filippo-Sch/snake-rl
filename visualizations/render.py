"""Portrait video and cover image rendered from the verified replay frames."""

from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1080, 1350, 30
BG, PANEL, EDGE = "#08151e", "#102832", "#284553"
WHITE, MUTED, FULL, PARTIAL, FRUIT = (
    "#edf8f8",
    "#a3bbc7",
    "#67e8c0",
    "#bca6ff",
    "#ff8074",
)
PALETTE = ["#284553", "#12313d", "#12313d", "#4ecb9c", "#b8ffe5"]
SCENES = [("intro", 3), ("b7", 13), ("b11", 13), ("classic", 15), ("outro", 5)]


@lru_cache(maxsize=None)
def font(size, bold=False):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    path = windows / ("segoeuib.ttf" if bold else "segoeui.ttf")
    if not path.exists():
        import matplotlib

        path = (
            Path(matplotlib.__file__).parent
            / "mpl-data/fonts/ttf"
            / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        )
    return ImageFont.truetype(str(path), size)


def text(draw, position, value, size=26, color=WHITE, bold=False):
    draw.text(position, str(value), font=font(size, bold), fill=color)


def dim(color):
    values = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    background = [8, 21, 30]
    return tuple(int(0.23 * value + 0.77 * bg) for value, bg in zip(values, background))


def board(draw, episode, frame, x, y, width, fog=False):
    size = episode["size"]
    unit = width / size
    head = frame["head"]
    for row in range(size):
        for col in range(size):
            value = int(frame["board"][row * size + col])
            hidden = fog and (abs(row - head[0]) > 2 or abs(col - head[1]) > 2)
            left, top = x + col * unit, y + (size - 1 - row) * unit
            gap, radius = max(1.5, unit * 0.045), max(3, unit * 0.16)
            color = dim(PALETTE[value]) if hidden else PALETTE[value]
            draw.rounded_rectangle(
                (left + gap, top + gap, left + unit - gap, top + unit - gap),
                radius=radius,
                fill=color,
            )
            if value == 2:
                apple = dim(FRUIT) if hidden else FRUIT
                draw.ellipse(
                    (
                        left + unit * 0.24,
                        top + unit * 0.29,
                        left + unit * 0.77,
                        top + unit * 0.80,
                    ),
                    fill=apple,
                )
                leaf = dim(FULL) if hidden else FULL
                draw.line(
                    (
                        left + unit * 0.5,
                        top + unit * 0.34,
                        left + unit * 0.55,
                        top + unit * 0.20,
                    ),
                    fill=leaf,
                    width=max(2, int(unit * 0.045)),
                )
                draw.ellipse(
                    (
                        left + unit * 0.54,
                        top + unit * 0.18,
                        left + unit * 0.72,
                        top + unit * 0.29,
                    ),
                    fill=leaf,
                )
                if not hidden:
                    draw.ellipse(
                        (
                            left + unit * 0.34,
                            top + unit * 0.38,
                            left + unit * 0.44,
                            top + unit * 0.48,
                        ),
                        fill="#ffd4c7",
                    )
            if value == 4:
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[frame["action"] or 0]
                for side in (-1, 1):
                    cx = left + unit * (0.5 + 0.18 * dx + 0.12 * side * abs(dy))
                    cy = top + unit * (0.5 + 0.18 * dy + 0.12 * side * abs(dx))
                    r = unit * 0.065
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#104c3a")
    if fog:
        lo_r, hi_r = max(0, head[0] - 2), min(size - 1, head[0] + 2)
        lo_c, hi_c = max(0, head[1] - 2), min(size - 1, head[1] + 2)
        draw.rounded_rectangle(
            (
                x + lo_c * unit,
                y + (size - 1 - hi_r) * unit,
                x + (hi_c + 1) * unit,
                y + (size - lo_r) * unit,
            ),
            radius=5,
            outline=PARTIAL,
            width=3,
        )


@lru_cache(maxsize=3)
def base(case):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        strength = max(0, 1 - y / 950)
        draw.line(
            (0, y, WIDTH, y),
            fill=(
                int(8 + 8 * strength),
                int(21 + 16 * strength),
                int(30 + 17 * strength),
            ),
        )
    text(draw, (48, 38), "SNAKE  /  DEEP Q-LEARNING", 22, FULL, True)
    text(draw, (44, 79), "Learning to play Snake", 58, WHITE, True)
    titles = {
        "b7": "01 / Original task  ·  7 × 7",
        "b11": "02 / Original task  ·  11 × 11",
        "classic": "03 / Classic Snake  ·  fatal collisions",
    }
    text(draw, (48, 167), titles[case], 31, WHITE, True)
    description = (
        "Nonfatal collisions · fixed 1,000-action horizon"
        if case != "classic"
        else "5 × 5 playable interior · complete the board to win"
    )
    text(draw, (48, 218), description, 26, MUTED)
    for index, view in enumerate(("FULL OBSERVATION", "PARTIAL OBSERVATION")):
        x = 44 + index * 512
        accent = FULL if index == 0 else PARTIAL
        draw.rounded_rectangle(
            (x, 286, x + 480, 984), radius=24, fill=PANEL, outline=EDGE, width=2
        )
        draw.line((x + 26, 289, x + 454, 289), fill=accent, width=4)
        text(draw, (x + 24, 318), view, 25, accent, True)
    draw.rounded_rectangle((44, 1010, 1036, 1200), radius=22, fill="#112b37")
    draw.line((47, 1038, 47, 1170), fill=PARTIAL, width=4)
    notes = {
        "b7": (
            "Seeing less means searching first.",
            "Partial test return: DQN 122.13 vs Direct 92.81.",
            "500 held-out episodes · one training seed",
        ),
        "b11": (
            "Exploration changes the outcome.",
            "Partial test return: DQN 32.88 vs Direct 16.92.",
            "Full vision: Direct 75.63 vs DQN 71.95.",
        ),
        "classic": (
            "Full vision wins. Partial vision cycles.",
            "Full: 283 / 500 wins. Partial: 4 / 500 wins.",
            "This clip shows test game #0 in both views.",
        ),
    }
    title, line, detail = notes[case]
    text(draw, (72, 1035), title, 35, WHITE, True)
    text(draw, (72, 1101), line, 29, MUTED)
    text(draw, (72, 1146), detail, 25, MUTED)
    text(
        draw,
        (48, 1232),
        "Real checkpoint replays · test episode #0 · seed 0",
        24,
        MUTED,
    )
    text(
        draw,
        (48, 1275),
        "Filippo Schiabel  ·  github.com/Filippo-Sch/snake-rl",
        24,
        WHITE,
    )
    return image


def paired(data, case, step, chapter_progress=0):
    image = base(case).copy()
    draw = ImageDraw.Draw(image)
    episodes = {e["id"]: e for e in data["episodes"]}
    for index, regime in enumerate(("full", "partial")):
        episode = episodes[case + "_" + regime]
        frame = episode["frames"][min(step, len(episode["frames"]) - 1)]
        x = 44 + 512 * index
        accent = FULL if index == 0 else PARTIAL
        action = (
            ["UP", "RIGHT", "DOWN", "LEFT"][frame["action"]]
            if frame["action"] is not None
            else "DONE"
        )
        text(draw, (x + 24, 363), f"Action {frame['t']}  /  {action}", 23, MUTED)
        board(draw, episode, frame, x + 24, 410, 432, fog=index == 1)
        if case == "classic":
            stats = (
                ("FRUIT", f"{frame['fruits']}/22"),
                ("LENGTH", f"{frame['length']}/25"),
            )
        else:
            stats = (
                ("FRUIT", frame["fruits"]),
                ("WALL HITS", frame["walls"]),
                ("BODY CUTS", frame["hits"]),
            )
        for j, (label, value) in enumerate(stats):
            sx = x + 24 + j * (144 if case != "classic" else 224)
            text(draw, (sx, 867), label, 19, MUTED, True)
            text(draw, (sx, 892), value, 43, WHITE, True)
        badge = (
            "BOARD COMPLETE"
            if frame["outcome"] == "win"
            else (
                "STATE CYCLE"
                if frame.get("cycle")
                else "5 × 5 AGENT VIEW" if index else "ENTIRE BOARD VISIBLE"
            )
        )
        text(draw, (x + 24, 954), badge, 18, accent, True)
    draw.rectangle((44, 1320, 1036, 1324), fill=EDGE)
    draw.rectangle((44, 1320, 44 + 992 * chapter_progress, 1324), fill=FULL)
    return image


def intro(data, tick=0):
    image = paired(data, "b7", tick)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 273), fill=BG)
    text(draw, (48, 35), "REINFORCEMENT LEARNING", 23, FULL, True)
    text(draw, (44, 80), "Same algorithm.", 67, WHITE, True)
    text(draw, (44, 156), "Different vision.", 67, PARTIAL, True)
    return image


def outro(data):
    image = paired(data, "classic", 180, 1)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 273), fill=BG)
    text(draw, (48, 35), "LEARNED POLICIES. REAL BEHAVIOUR.", 23, FULL, True)
    text(draw, (44, 80), "The rules change", 62, WHITE, True)
    text(draw, (44, 154), "what learning solves.", 62, PARTIAL, True)
    return image


def poster(data, output):
    intro(data, 18).save(output)


def video(data, output, executable):
    binary = shutil.which(executable) or (
        str(Path(executable)) if Path(executable).is_file() else None
    )
    if not binary:
        raise FileNotFoundError(
            "FFmpeg is required only for video export; pass --ffmpeg PATH or install it on PATH"
        )
    log = output.with_suffix(".log")
    command = [
        binary,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "4",
        "-movflags",
        "+faststart",
        "-metadata",
        "title=Learning to Play Snake",
        str(output),
    ]
    with log.open("w", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            for scene, duration in SCENES:
                print(f"Rendering {scene}: {duration} seconds...", flush=True)
                for number in range(duration * FPS):
                    seconds = number / FPS
                    if scene == "intro":
                        image = intro(data, int(seconds * 4))
                    elif scene == "outro":
                        image = outro(data)
                    else:
                        rate = 16 if scene != "classic" else 12
                        image = paired(
                            data,
                            scene,
                            int(seconds * rate),
                            number / (duration * FPS - 1),
                        )
                    process.stdin.write(image.tobytes())
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
    if code:
        raise RuntimeError(log.read_text(encoding="utf-8"))
    print("Video:", output, flush=True)
