"""Reproduce the default bundle from the PIMA teaching notebook.

Install requirements-generation.txt into an environment with CPU PyTorch, then run
python scripts/generate_pima_bundle.py. Exercise answers are supplied in memory;
the student notebook remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DATA_SHA256 = "698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional local copy of the notebook's diabetes.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "pima")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    data = (
        args.dataset.read_bytes()
        if args.dataset
        else urllib.request.urlopen(DATA_URL, timeout=60).read()
    )
    digest = hashlib.sha256(data).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(
            "Dataset SHA-256 differs from the recorded PIMA source. Inspect the dataset before changing the expected hash."
        )
    import torch

    torch.set_num_threads(1)
    notebook = (
        ROOT / "reference" / "EdgeMind_ANN_Part2_PIMA_Predict_3FIXME_Gamified.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    replacements = {
        "self.fc1 = ______________________________": "self.fc1 = nn.Linear(8, 16)",
        "criterion = ______________________________": "criterion = nn.BCEWithLogitsLoss()",
        "optimizer = ______________________________": "optimizer = optim.Adam(model.parameters(), lr=0.01)",
    }
    namespace = {"__name__": "__pima_generation__", "display": lambda *_: None}
    previous_dir = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="pima-generation-") as directory:
            os.chdir(directory)
            Path("diabetes.csv").write_bytes(data)
            for cell in cells:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                for exercise, answer in replacements.items():
                    source = source.replace(exercise, answer)
                # Execute only the local teaching notebook chosen above.
                exec(compile(source, str(notebook), "exec"), namespace)  # noqa: S102
            source_dir = Path("deployment/pima")
            config = json.loads((source_dir / "model.json").read_text())
            config["training"] = {
                "seed": 42,
                "test_split": 0.2,
                "stratified": True,
                "epochs": 100,
                "optimizer": "Adam",
                "learning_rate": 0.01,
                "loss": "BCEWithLogitsLoss",
                "device": str(namespace["device"]),
                "dataset_url": DATA_URL,
                "dataset_sha256": digest,
            }
            config["metrics"] = {"test_accuracy": float(namespace["accuracy"])}
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.pt").write_bytes(
                (source_dir / "model.pt").read_bytes()
            )
            (output_dir / "model.json").write_text(
                json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                f"Default bundle written to {output_dir}; test accuracy: {namespace['accuracy']:.4f}"
            )
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
