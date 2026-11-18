"""Serve the two-file ANN bundles exported by the teaching notebooks."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn

from app.worker import resolve_device


class AnnConfigurationError(ValueError):
    """A registered bundle cannot be served until its files are corrected."""


def _reject_json_constant(token: str):
    raise ValueError(f"Nonstandard JSON number: {token}")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, ValueError) as error:
        raise AnnConfigurationError(
            f"Could not read {path.name}. Copy the complete export and restart the API and worker."
        ) from error
    if not isinstance(value, dict):
        raise AnnConfigurationError(f"{path.name} must contain a JSON object.")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnnConfigurationError(message)


def _finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_manifest(config: dict, model_id: str) -> None:
    """Validate the subset of the export schema implemented by this runtime."""
    try:
        _require(
            config["schema_version"] == 1 and config["framework"] == "pytorch",
            "Use a schema_version 1 PyTorch notebook export.",
        )
        _require(
            config["model_id"] == model_id,
            "The folder name must match model_id in model.json.",
        )
        _require(
            isinstance(config["display_name"], str) and bool(config["display_name"]),
            "display_name must be nonempty.",
        )
        _require(
            isinstance(config["model_version"], str) and bool(config["model_version"]),
            "model_version must be a nonempty string.",
        )
        _require(
            config["weights"] == {"file": "model.pt", "format": "state_dict"},
            "Export weights as a state_dict in the neighboring model.pt file.",
        )
        inputs, output = config["input"], config["output"]
        shape = inputs["shape"]
        _require(
            inputs["dtype"] == "float32"
            and isinstance(shape, list)
            and shape[0] is None,
            "Input must be batched float32 with a null batch dimension.",
        )
        _require(
            all(type(size) is int and size > 0 for size in shape[1:]),
            "Input dimensions must be positive integers.",
        )
        pre = inputs["preprocessing"]
        if inputs["type"] == "tabular":
            features = inputs["features"]
            _require(
                len(shape) == 2
                and isinstance(features, list)
                and len(features) == shape[1],
                "Feature count must match input shape.",
            )
            _require(
                all(isinstance(name, str) and name for name in features)
                and len(set(features)) == len(features),
                "Feature names must be nonempty and unique.",
            )
            _require(
                pre["type"] in {"none", "standard_scaler"},
                "Unsupported numeric preprocessing.",
            )
            if pre["type"] == "standard_scaler":
                _require(
                    len(pre["mean"]) == len(features)
                    and len(pre["scale"]) == len(features),
                    "Scaler values must match feature count.",
                )
                _require(
                    all(_finite(x) for x in pre["mean"])
                    and all(_finite(x) and x > 0 for x in pre["scale"]),
                    "Scaler means must be finite and scales must be positive and finite.",
                )
        else:
            _require(
                inputs["type"] == "image"
                and len(shape) == 4
                and shape[1] == 1
                and inputs["layout"] == "NCHW",
                "Images must use shape [null, 1, height, width] and NCHW layout.",
            )
            _require(
                pre["type"] == "image"
                and pre["color_mode"] == "L"
                and pre["resize"] == shape[2:],
                "Image preprocessing must specify grayscale and the configured height/width.",
            )
            _require(
                pre["resize_interpolation"] == "bilinear"
                and _finite(pre["pixel_scale"])
                and pre["pixel_scale"] > 0
                and type(pre["invert"]) is bool,
                "Use bilinear resizing, a positive pixel scale, and a boolean invert setting.",
            )
        architecture = config["architecture"]
        layers = architecture["layers"]
        _require(
            architecture["type"] == "sequential"
            and isinstance(layers, list)
            and bool(layers),
            "Export a nonempty sequential architecture.",
        )
        dimensions = shape[1:]
        parameters = 0
        for layer in layers:
            kind = layer["type"]
            if kind == "flatten":
                _require(
                    layer["start_dim"] == 1 and layer["end_dim"] == -1,
                    "Only flattening all non-batch dimensions is supported.",
                )
                dimensions = [math.prod(dimensions)]
            elif kind == "linear":
                _require(
                    type(layer["in_features"]) is int
                    and type(layer["out_features"]) is int
                    and layer["out_features"] > 0
                    and type(layer["bias"]) is bool,
                    "Linear dimensions must be positive integers and bias must be boolean.",
                )
                _require(
                    dimensions == [layer["in_features"]],
                    "Linear input dimensions do not match the preceding layer. Include flatten for images.",
                )
                parameters += layer["in_features"] * layer["out_features"] + (
                    layer["out_features"] if layer["bias"] else 0
                )
                dimensions = [layer["out_features"]]
            else:
                _require(kind == "relu", f"Unsupported layer type: {kind}.")
        _require(
            parameters <= 10_000_000,
            "This teaching server supports ANN bundles with at most 10 million parameters.",
        )
        _require(
            len(dimensions) == 1 and output["shape"] == [None, dimensions[0]],
            "Output shape must match the network.",
        )
        _require(
            isinstance(output["target"], str) and bool(output["target"]),
            "Output target must be a nonempty string.",
        )
        labels, post = output["labels"], output["postprocessing"]
        _require(
            isinstance(labels, list)
            and all(isinstance(x, str) and x for x in labels)
            and len(set(labels)) == len(labels),
            "Labels must be unique nonempty strings in class-index order.",
        )
        task = output["task"]
        if task == "binary_classification":
            _require(
                dimensions == [1]
                and len(labels) == 2
                and post["type"] == "sigmoid"
                and post["positive_class_index"] == 1
                and _finite(post["threshold"])
                and 0 <= post["threshold"] <= 1,
                "Binary models require one logit, two labels, sigmoid, positive class index 1, and a threshold from 0 to 1.",
            )
        elif task == "multiclass_classification":
            _require(
                dimensions[0] >= 2
                and len(labels) == dimensions[0]
                and post == {"type": "softmax", "dim": 1},
                "Multiclass labels must match logits; use softmax along dimension 1.",
            )
        else:
            _require(
                task == "regression"
                and dimensions == [1]
                and labels == []
                and post == {"type": "none"},
                "Regression requires one output with no class labels or postprocessing.",
            )
        example = config["example"]
        _require(isinstance(example, dict), "Include an example input in model.json.")
        if inputs["type"] == "tabular":
            values = example["features"]
            _require(
                isinstance(values, dict)
                and set(values) == set(inputs["features"])
                and all(_finite(value) for value in values.values()),
                "The example must contain finite raw values for every feature.",
            )
        else:
            image = example["image"]
            pixels = image["pixels"]
            height, width = inputs["shape"][2:]
            _require(
                image["encoding"] == "uint8_pixels"
                and image["shape"] == [height, width]
                and isinstance(pixels, list)
                and len(pixels) == height
                and all(
                    isinstance(row, list)
                    and len(row) == width
                    and all(type(value) is int and 0 <= value <= 255 for value in row)
                    for row in pixels
                ),
                "The example image must contain a height-by-width array of byte pixels.",
            )
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as error:
        raise AnnConfigurationError(
            f"Incomplete or invalid model.json ({error}). Run the notebook export cell again."
        ) from error


def build_network(config: dict) -> nn.Sequential:
    layers = []
    for layer in config["architecture"]["layers"]:
        kind = layer["type"]
        if kind == "linear":
            layers.append(
                nn.Linear(
                    layer["in_features"], layer["out_features"], bias=layer["bias"]
                )
            )
        elif kind == "flatten":
            layers.append(
                nn.Flatten(start_dim=layer["start_dim"], end_dim=layer["end_dim"])
            )
        else:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Registry:
    def __init__(self, config_path: Path, model_root: Path):
        self.root = model_root.resolve()
        self.models: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
        self.loaded: dict[str, nn.Module] = {}
        self.default_model = None
        self.configuration_error = None
        try:
            registration = _read_json(config_path)
            ids = registration["models"]
            self.default_model = registration["default_model"]
            _require(
                isinstance(ids, list)
                and all(isinstance(x, str) for x in ids)
                and len(set(ids)) == len(ids),
                "models must be a list of unique folder names.",
            )
            _require(
                isinstance(self.default_model, str) and self.default_model in ids,
                "default_model must name a registered model.",
            )
        except (AnnConfigurationError, KeyError, TypeError) as error:
            self.configuration_error = f"Invalid ANN registration: {error}"
            return
        for model_id in ids:
            try:
                _require(
                    bool(re.fullmatch(r"[a-zA-Z0-9_-]+", model_id)),
                    "Use a simple folder name containing letters, numbers, underscores, or hyphens.",
                )
                folder = (self.root / model_id).resolve()
                _require(
                    folder.is_relative_to(self.root),
                    "Model folder must remain inside ANN_MODEL_DIR.",
                )
                _require(
                    (folder / "model.json").resolve().is_relative_to(self.root),
                    "Model files must remain inside ANN_MODEL_DIR.",
                )
                config = _read_json(folder / "model.json")
                validate_manifest(config, model_id)
                weights = (folder / "model.pt").resolve()
                _require(
                    weights.is_relative_to(self.root) and weights.is_file(),
                    "Missing model.pt. Copy both exported files into the model folder.",
                )
                self.models[model_id] = config
            except AnnConfigurationError as error:
                self.errors[model_id] = str(error)

    def catalog(self) -> dict:
        return {
            "default_model": self.default_model,
            "models": list(self.models.values()),
            "errors": self.errors,
            "configuration_error": self.configuration_error,
        }

    def get(self, model_id: str) -> dict:
        if self.configuration_error:
            raise AnnConfigurationError(self.configuration_error)
        if model_id in self.errors:
            raise AnnConfigurationError(f"{model_id}: {self.errors[model_id]}")
        if model_id not in self.models:
            raise KeyError(model_id)
        return self.models[model_id]

    def network(self, model_id: str) -> nn.Module:
        config = self.get(model_id)
        if model_id not in self.loaded:
            try:
                model = build_network(config)
                state = torch.load(
                    self.root / model_id / "model.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(state, strict=True)
            except Exception as error:
                raise AnnConfigurationError(
                    f"{model_id}: model.pt does not match model.json. Copy both files from the same notebook export."
                ) from error
            self.loaded[model_id] = model.to(resolve_device()).eval()
        return self.loaded[model_id]


@lru_cache(maxsize=1)
def registry() -> Registry:
    return Registry(
        Path(os.environ.get("ANN_CONFIG", "config/ann.json")),
        Path(os.environ.get("ANN_MODEL_DIR", "models")),
    )


def numeric_input(config: dict, features: dict) -> torch.Tensor:
    if config["input"]["type"] != "tabular":
        raise ValueError("This model requires an image upload.")
    names = config["input"]["features"]
    if set(features) != set(names):
        raise ValueError(
            f"Features must match the model. Missing: {sorted(set(names) - set(features))}; extra: {sorted(set(features) - set(names))}."
        )
    if not all(_finite(features[name]) for name in names):
        raise ValueError("Every feature must be a finite number.")
    # float64 preserves the fitted sklearn transform before conversion to float32.
    values = torch.tensor([[features[name] for name in names]], dtype=torch.float64)
    pre = config["input"]["preprocessing"]
    if pre["type"] == "standard_scaler":
        values = (
            values - torch.tensor(pre["mean"], dtype=torch.float64)
        ) / torch.tensor(pre["scale"], dtype=torch.float64)
    values = values.float()
    if not torch.isfinite(values).all():
        raise ValueError(
            "Feature values are outside the model's supported numeric range."
        )
    return values


def image_input(config: dict, path: Path) -> tuple[torch.Tensor, dict]:
    if config["input"]["type"] != "image":
        raise ValueError("This model requires numeric features.")
    pre = config["input"]["preprocessing"]
    height, width = pre["resize"]
    with Image.open(path) as opened:
        image = (
            ImageOps.exif_transpose(opened)
            .convert("L")
            .resize((width, height), Image.Resampling.BILINEAR)
        )
    if pre["invert"]:
        image = ImageOps.invert(image)
    tensor = (
        torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(
            1, 1, height, width
        )
        * pre["pixel_scale"]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    preview = {
        "width": width,
        "height": height,
        "data_url": "data:image/png;base64,"
        + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    return tensor, preview


def _predict(
    model_id: str, features: dict | None = None, image_filename: str | None = None
) -> dict:
    """RQ inference task, sharing the existing inference queue and device setting."""
    catalog = registry()
    try:
        config = catalog.get(model_id)
    except KeyError as error:
        raise AnnConfigurationError(
            f"{model_id} is not registered in the worker. Restart the API and worker together after editing config/ann.json."
        ) from error
    preview = None
    if image_filename is not None:
        tensor, preview = image_input(
            config, Path(os.environ.get("UPLOAD_DIR", "data/uploads")) / image_filename
        )
    else:
        tensor = numeric_input(config, features or {})
    model = catalog.network(model_id)
    device = next(model.parameters()).device
    with torch.inference_mode():
        output = model(tensor.to(device)).cpu()
    if (
        list(output.shape) != [1, config["output"]["shape"][1]]
        or not torch.isfinite(output).all()
    ):
        raise AnnConfigurationError(
            "Model produced invalid output. Check the notebook and export again."
        )
    settings = config["output"]
    result = {
        "model_id": model_id,
        "model_version": config["model_version"],
        "target": settings["target"],
        "task": settings["task"],
    }
    if settings["task"] == "regression":
        result["prediction"] = float(output.item())
    else:
        labels = settings["labels"]
        if settings["task"] == "binary_classification":
            probability = float(output.sigmoid().item())
            probabilities = [1 - probability, probability]
            index = int(probability >= settings["postprocessing"]["threshold"])
        else:
            probabilities = output.softmax(1)[0].tolist()
            index = int(output.argmax(1).item())
        result.update(
            prediction=labels[index], probabilities=dict(zip(labels, probabilities))
        )
    if preview:
        result["image_preview"] = preview
    return result


def predict(
    model_id: str, features: dict | None = None, image_filename: str | None = None
) -> dict:
    """Keep expected ANN failure messages available to the job polling UI."""
    from rq import get_current_job

    try:
        return _predict(model_id, features, image_filename)
    except (AnnConfigurationError, ValueError, FileNotFoundError) as error:
        job = get_current_job()
        if job is not None:
            job.meta["ann_error"] = str(error)
            job.save_meta()
        raise
