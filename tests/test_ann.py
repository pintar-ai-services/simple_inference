import copy
import io
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from redis.exceptions import ConnectionError as RedisConnectionError

from app import ann, main


def bundle_config(model_id="iris", task="multiclass_classification"):
    image = model_id == "mnist"
    input_size, output_size = (784, 10) if image else (4, 3)
    if task != "multiclass_classification":
        output_size = 1
    layers = [
        {
            "type": "linear",
            "in_features": input_size,
            "out_features": output_size,
            "bias": True,
        }
    ]
    inputs = {
        "type": "tabular",
        "dtype": "float32",
        "shape": [None, input_size],
        "features": [f"feature_{i}" for i in range(input_size)],
        "preprocessing": {"type": "none"},
    }
    example = {"features": dict.fromkeys(inputs["features"], 1.0)}
    if image:
        layers.insert(0, {"type": "flatten", "start_dim": 1, "end_dim": -1})
        inputs = {
            "type": "image",
            "dtype": "float32",
            "shape": [None, 1, 28, 28],
            "layout": "NCHW",
            "preprocessing": {
                "type": "image",
                "color_mode": "L",
                "resize": [28, 28],
                "resize_interpolation": "bilinear",
                "pixel_scale": 1 / 255,
                "invert": False,
            },
        }
        example = {
            "image": {
                "encoding": "uint8_pixels",
                "shape": [28, 28],
                "pixels": [[0] * 28 for _ in range(28)],
            }
        }
    labels = [str(i) for i in range(output_size)]
    post = {"type": "softmax", "dim": 1}
    if task == "binary_classification":
        labels, post = (
            ["negative", "positive"],
            {"type": "sigmoid", "positive_class_index": 1, "threshold": 0.5},
        )
    elif task == "regression":
        labels, post = [], {"type": "none"}
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_version": "1",
        "framework": "pytorch",
        "display_name": model_id.title(),
        "weights": {"file": "model.pt", "format": "state_dict"},
        "architecture": {"type": "sequential", "layers": layers},
        "input": inputs,
        "output": {
            "task": task,
            "shape": [None, output_size],
            "target": "result",
            "labels": labels,
            "postprocessing": post,
        },
        "example": example,
    }


def write_bundle(root, config):
    folder = root / config["model_id"]
    folder.mkdir(parents=True)
    (folder / "model.json").write_text(json.dumps(config))
    network = ann.build_network(config)
    # Predictable class scores make label order and threshold errors visible.
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network[-1].bias.copy_(
            torch.arange(network[-1].out_features, dtype=torch.float32)
        )
    torch.save(network.state_dict(), folder / "model.pt")
    return network


@pytest.fixture
def registered(monkeypatch, tmp_path):
    configs = [
        bundle_config(),
        bundle_config("mnist"),
        bundle_config("concept", "regression"),
        bundle_config("binary", "binary_classification"),
    ]
    for config in configs:
        write_bundle(tmp_path, config)
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps(
            {"default_model": "iris", "models": [c["model_id"] for c in configs]}
        )
    )
    registry = ann.Registry(path, tmp_path)
    assert not registry.errors
    monkeypatch.setattr(ann, "registry", lambda: registry)
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main, "upload_dir", tmp_path)
    return registry


def test_bundled_pima_and_scaler_match_notebook_math(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    registry = ann.Registry(Path("config/ann.json"), Path("models"))
    monkeypatch.setattr(ann, "registry", lambda: registry)
    assert registry.default_model == "pima" and registry.errors == {}
    config = registry.get("pima")
    features = config["example"]["features"]
    reversed_features = dict(reversed(list(features.items())))
    tensor = ann.numeric_input(config, reversed_features)
    pre = config["input"]["preprocessing"]
    expected = [
        (features[name] - mean) / scale
        for name, mean, scale in zip(
            config["input"]["features"], pre["mean"], pre["scale"]
        )
    ]
    torch.testing.assert_close(tensor, torch.tensor([expected]))
    result = ann.predict("pima", features)
    probability = float(registry.network("pima")(tensor).sigmoid().item())
    assert result["probabilities"]["Diabetes"] == pytest.approx(probability)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["prediction"] == config["output"]["labels"][int(probability >= 0.5)]
    assert 0 <= config["metrics"]["test_accuracy"] <= 1
    assert len(config["training"]["dataset_sha256"]) == 64


def test_multiclass_regression_and_binary_threshold(registered):
    result = ann.predict("iris", registered.get("iris")["example"]["features"])
    assert result["prediction"] == "2" and sum(
        result["probabilities"].values()
    ) == pytest.approx(1)
    assert registered.network("iris") is registered.network("iris")
    regression = ann.predict(
        "concept", registered.get("concept")["example"]["features"]
    )
    assert regression["prediction"] == 0 and "probabilities" not in regression
    binary = ann.predict("binary", registered.get("binary")["example"]["features"])
    assert binary["prediction"] == "positive"  # sigmoid(0) == threshold, so use >=.
    assert binary["probabilities"] == {"negative": 0.5, "positive": 0.5}


@pytest.mark.parametrize("bad", [None, True, "1", float("inf"), float("nan"), 10**1000])
def test_numeric_input_rejects_nonfinite_or_nonnumeric_values(registered, bad):
    config = registered.get("iris")
    features = dict(config["example"]["features"], feature_0=bad)
    with pytest.raises(ValueError, match="finite number"):
        ann.numeric_input(config, features)


def test_feature_mismatch_and_input_kind(registered):
    with pytest.raises(ValueError, match="Missing"):
        ann.numeric_input(registered.get("iris"), {"extra": 1})
    with pytest.raises(ValueError, match="image upload"):
        ann.numeric_input(registered.get("mnist"), {})


def test_image_preprocessing_and_exact_preview(registered, tmp_path):
    path = tmp_path / "digit.png"
    Image.new("RGB", (56, 56), (255, 255, 255)).save(path)
    tensor, preview = ann.image_input(registered.get("mnist"), path)
    assert tensor.shape == (1, 1, 28, 28)
    torch.testing.assert_close(tensor, torch.ones_like(tensor))
    assert preview["width"] == preview["height"] == 28
    assert preview["data_url"].startswith("data:image/png;base64,")
    result = ann.predict("mnist", image_filename=path.name)
    assert result["prediction"] == "9" and result["image_preview"] == preview
    inverted = copy.deepcopy(registered.get("mnist"))
    inverted["input"]["preprocessing"]["invert"] = True
    tensor, _ = ann.image_input(inverted, path)
    assert tensor.count_nonzero() == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(schema_version=2),
        lambda c: c["architecture"]["layers"][0].update(type="custom_python"),
        lambda c: c["architecture"]["layers"][0].update(in_features=5),
        lambda c: c["output"].update(labels=["one"]),
        lambda c: c["weights"].update(file="../outside.pt"),
        lambda c: c["input"].update(
            preprocessing={"type": "standard_scaler", "mean": [0] * 4, "scale": [0] * 4}
        ),
        lambda c: c["input"].update(shape=[]),
        lambda c: c.update(input=None),
    ],
)
def test_invalid_manifests_are_actionable(mutation):
    config = bundle_config()
    mutation(config)
    with pytest.raises(ann.AnnConfigurationError):
        ann.validate_manifest(config, "iris")


def test_missing_and_mismatched_weights(registered, tmp_path):
    (tmp_path / "iris" / "model.pt").unlink()
    registration = tmp_path / "ann.json"
    broken = ann.Registry(registration, tmp_path)
    assert "Missing model.pt" in broken.errors["iris"]
    assert broken.default_model == "iris"  # No fallback to another registered model.
    with pytest.raises(ann.AnnConfigurationError, match="iris"):
        broken.get("iris")
    torch.save({"bad": torch.zeros(1)}, tmp_path / "iris" / "model.pt")
    with pytest.raises(ann.AnnConfigurationError, match="same notebook export"):
        registered.network("iris")


def test_bad_registration_and_escape_paths(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text('{"default_model":"../outside","models":["../outside"]}')
    registry = ann.Registry(path, tmp_path / "models")
    assert "../outside" in registry.errors
    path.write_text('{"default_model":"missing","models":[]}')
    assert ann.Registry(path, tmp_path).configuration_error
    assert ann.Registry(tmp_path / "absent.json", tmp_path).configuration_error


def test_symlink_cannot_escape_model_root(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    write_bundle(outside, bundle_config())
    (root / "iris").symlink_to(outside / "iris", target_is_directory=True)
    path = tmp_path / "ann.json"
    path.write_text('{"default_model":"iris","models":["iris"]}')
    assert "inside ANN_MODEL_DIR" in ann.Registry(path, root).errors["iris"]


def test_ann_api_queues_numeric_and_image_jobs(registered, monkeypatch):
    queued = []
    monkeypatch.setattr(main, "has_inference_worker", lambda _: True)
    monkeypatch.setattr(
        main.queue, "enqueue", lambda *args, **kwargs: queued.append((args, kwargs))
    )
    with TestClient(main.app) as client:
        catalog = client.get("/ann/models").json()
        assert catalog["default_model"] == "iris" and len(catalog["models"]) == 4
        features = registered.get("iris")["example"]["features"]
        response = client.post("/ann/models/iris/jobs", json={"features": features})
        assert response.status_code == 202
        assert queued[-1][0] == (ann.predict, "iris")
        assert queued[-1][1]["features"] == features
        image = io.BytesIO()
        Image.new("L", (28, 28), 255).save(image, format="PNG")
        response = client.post(
            "/ann/models/mnist/image-jobs",
            files={"image": ("digit.png", image.getvalue(), "image/png")},
        )
        assert response.status_code == 202
        filename = queued[-1][1]["image_filename"]
        assert (main.upload_dir / filename).exists()
        assert (
            client.post(
                "/ann/models/mnist/jobs", json={"features": features}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/ann/models/missing/jobs", json={"features": features}
            ).status_code
            == 404
        )
        assert (
            client.post("/ann/models/iris/jobs", json={"features": {}}).status_code
            == 400
        )
        assert client.post("/tabular/training-jobs", json={}).status_code == 404


def test_ann_api_worker_and_configuration_failures(registered, monkeypatch):
    client = TestClient(main.app)
    features = registered.get("iris")["example"]["features"]
    monkeypatch.setattr(main, "has_inference_worker", lambda _: False)
    assert (
        client.post("/ann/models/iris/jobs", json={"features": features}).status_code
        == 503
    )
    monkeypatch.setattr(main, "has_inference_worker", lambda _: True)
    monkeypatch.setattr(
        main.queue,
        "enqueue",
        lambda *a, **k: (_ for _ in ()).throw(RedisConnectionError("down")),
    )
    assert (
        client.post("/ann/models/iris/jobs", json={"features": features}).status_code
        == 503
    )
    registered.errors["iris"] = "Missing model.pt. Copy both exported files."
    response = client.post("/ann/models/iris/jobs", json={"features": features})
    assert response.status_code == 503 and "Copy both" in response.json()["detail"]


def test_ann_failure_message_survives_job_polling(registered, monkeypatch):
    class FakeJob:
        id = "ann-failed"

        def __init__(self):
            self.meta = {}

        def save_meta(self):
            pass

        def get_status(self, refresh=False):
            return "failed"

    job = FakeJob()
    monkeypatch.setattr("rq.get_current_job", lambda: job)
    with pytest.raises(ValueError, match="Missing"):
        ann.predict("iris", {})
    monkeypatch.setattr(main.Job, "fetch", lambda *_args, **_kwargs: job)
    response = TestClient(main.app).get("/jobs/ann-failed")
    assert response.status_code == 200
    assert "Missing" in response.json()["error"]


def test_image_queue_failure_removes_staged_upload(registered, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "has_inference_worker", lambda _: True)
    monkeypatch.setattr(
        main.queue,
        "enqueue",
        lambda *a, **k: (_ for _ in ()).throw(RedisConnectionError("down")),
    )
    image = io.BytesIO()
    Image.new("L", (28, 28)).save(image, format="PNG")
    response = TestClient(main.app).post(
        "/ann/models/mnist/image-jobs",
        files={"image": ("digit.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 503
    assert list(tmp_path.glob("*.png")) == []


def test_invalid_example_is_a_configuration_error():
    config = bundle_config("mnist")
    config["example"]["image"]["pixels"][0][0] = 256
    with pytest.raises(ann.AnnConfigurationError, match="byte pixels"):
        ann.validate_manifest(config, "mnist")


@pytest.mark.parametrize(
    "notebook_name,model_id",
    [
        ("EdgeMind_ANN_Part1_ANN_Concept_3FIXME_Gamified.ipynb", "ann_concept"),
        ("EdgeMind_ANN_Part2_PIMA_Predict_3FIXME_Gamified.ipynb", "pima"),
        ("EdgeMind_ANN_Part3_MNIST_Vision_3FIXME_Gamified.ipynb", "mnist"),
        ("Answer_Pytorch IRIS-ANN.ipynb", "iris"),
    ],
)
def test_actual_notebook_export_is_servable(
    notebook_name, model_id, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from torch import nn

    cells = json.loads((Path("reference") / notebook_name).read_text())["cells"]
    export = next(
        "".join(c["source"]) for c in cells if c.get("id") == f"export-{model_id}-code"
    )
    namespace = {"torch": torch, "nn": nn}
    if model_id == "ann_concept":
        namespace.update(
            model=nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)),
            x=torch.tensor([[1.0, 2.0, 3.0]]),
        )
    elif model_id == "iris":
        source = next(
            "".join(c["source"])
            for c in cells
            if "class NeuralNet" in "".join(c["source"])
        )
        exec(source, namespace)  # noqa: S102 - local notebook under test.
        namespace.update(
            X_test=torch.ones(1, 4),
            label_encoder=SimpleNamespace(
                classes_=["setosa", "versicolor", "virginica"]
            ),
        )
    elif model_id == "mnist":
        source = next(
            "".join(c["source"])
            for c in cells
            if "class MNIST_ANN" in "".join(c["source"])
        )
        source = source.replace(
            "self.fc3 = ______________________________", "self.fc3 = nn.Linear(64, 10)"
        )
        namespace.update(
            device=torch.device("cpu"), sample_images=torch.ones(1, 1, 28, 28)
        )
        exec(source, namespace)  # noqa: S102 - local notebook under test.
    else:
        # The real PIMA export also exercises the dataframe/scaler interface.
        class Row(dict):
            pass

        class Columns(list):
            def tolist(self):
                return list(self)

        class Frame:
            def __init__(self):
                self.columns = Columns(f"feature_{i}" for i in range(8))
                self.iloc = [Row(dict.fromkeys(self.columns, 1.0))]

            def drop(self, *args, **kwargs):
                return self

        source = next(
            "".join(c["source"])
            for c in cells
            if "class PIMA_ANN" in "".join(c["source"])
        )
        source = source.replace(
            "self.fc1 = ______________________________", "self.fc1 = nn.Linear(8, 16)"
        )
        namespace.update(
            device=torch.device("cpu"),
            X_test_tensor=torch.ones(1, 8),
            dataset=Frame(),
            scaler=SimpleNamespace(mean_=torch.zeros(8), scale_=torch.ones(8)),
        )
        exec(source, namespace)  # noqa: S102 - local notebook under test.
    monkeypatch.chdir(tmp_path)
    exec(export, namespace)  # noqa: S102 - local notebook export under test.
    registration = tmp_path / "ann.json"
    registration.write_text(
        json.dumps({"default_model": model_id, "models": [model_id]})
    )
    registry = ann.Registry(registration, tmp_path / "deployment")
    assert not registry.errors
    monkeypatch.setattr(ann, "registry", lambda: registry)
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    config = registry.get(model_id)
    if model_id == "mnist":
        Image.new("L", (28, 28), 255).save(tmp_path / "example.png")
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
        result = ann.predict(model_id, image_filename="example.png")
    else:
        result = ann.predict(model_id, config["example"]["features"])
    assert result["model_id"] == model_id
