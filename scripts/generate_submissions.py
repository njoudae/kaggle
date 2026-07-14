from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import columns_from_config, labels_from_config
from src.inference import generate_submission_file, zip_submissions
from src.utils import ensure_dir, get_device, set_seed


def resolve_test_path(data_dir: Path, preferred: str, fallback: str | None, label: str) -> Path:
    preferred_path = data_dir / preferred
    if preferred_path.exists():
        return preferred_path
    if fallback:
        fallback_path = data_dir / fallback
        if fallback_path.exists():
            print(f"Official {label} test file not found: {preferred_path}")
            print(f"Using fallback {label} file from the current repository: {fallback_path}")
            return fallback_path
    raise FileNotFoundError(
        f"Could not find {label} test file. Tried {preferred_path}"
        + (f" and fallback {data_dir / fallback}" if fallback else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate StanceEval2026 submission files.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory. Defaults to paths.best_model_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    columns = columns_from_config(config)
    _, id2label = labels_from_config(config)
    data_dir = Path(config["paths"]["data_dir"])
    results_dir = ensure_dir(config["paths"]["results_dir"])
    checkpoint_dir = Path(args.checkpoint or config["paths"]["best_model_dir"])
    device = get_device()
    training = config["training"]
    seen_test_path = resolve_test_path(
        data_dir,
        config["paths"]["test_seen_file"],
        config["paths"].get("fallback_test_seen_file"),
        "seen",
    )
    unseen_test_path = resolve_test_path(
        data_dir,
        config["paths"]["test_unseen_file"],
        config["paths"].get("fallback_test_unseen_file"),
        "unseen",
    )

    seen = generate_submission_file(
        test_path=seen_test_path,
        output_path=results_dir / config["submissions"]["seen_name"],
        checkpoint_dir=checkpoint_dir,
        columns=columns,
        id2label=id2label,
        max_length=int(training["max_length"]),
        batch_size=int(training["eval_batch_size"]),
        device=device,
        num_workers=int(training.get("num_workers", 0) or 0),
    )
    unseen = generate_submission_file(
        test_path=unseen_test_path,
        output_path=results_dir / config["submissions"]["unseen_name"],
        checkpoint_dir=checkpoint_dir,
        columns=columns,
        id2label=id2label,
        max_length=int(training["max_length"]),
        batch_size=int(training["eval_batch_size"]),
        device=device,
        num_workers=int(training.get("num_workers", 0) or 0),
    )
    print(f"Saved: {seen}")
    print(f"Saved: {unseen}")

    if bool(config["submissions"].get("create_zip", True)):
        zip_path = zip_submissions(results_dir / config["submissions"]["zip_name"], [seen, unseen])
        print(f"Saved: {zip_path}")


if __name__ == "__main__":
    main()
