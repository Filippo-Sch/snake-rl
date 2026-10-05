import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("b7_full", "b7_partial", "b11_full", "b11_partial")
SELECTED = {
    "b7_full": "dqn_2048000",
    "b7_partial": "dqn_2048000",
    "b11_full": "dqn_C_selected",
    "b11_partial": "dqn_C_selected",
}
RUNS = {
    "scratch_full": "phase2/full/scratch",
    "scratch_partial": "phase2/partial/scratch",
    "extended_full": "phase2_extended/full/scratch",
    "extended_partial": "phase2_extended/partial/scratch",
    "full_control": "phase2_policy/full/control",
    "full_lower_lr": "phase2_policy/full/variant",
    "partial_protected": "phase2_policy/partial/variant",
    "full_watkins": "phase2_watkins/full/watkins",
    "full_discount": "phase2_discount/full/discount",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_csv(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(values)
    if not values:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label, tolerance=1e-8):
    require(
        abs(float(actual) - float(expected)) <= tolerance,
        f"{label}: expected {expected}, got {actual}",
    )


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(
        path != root and root in path.parents,
        f"Path outside data directory: {relative}",
    )
    return path
