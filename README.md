# TNB Inference

TNB Inference is a small asynchronous PyTorch demo with a Pascal VOC transmission-line image viewer, image classification, object detection, and configurable ANN inference.

```text
browser → FastAPI → Redis/RQ queue → worker → database/files → browser
```

## Run it

Prerequisites: Docker Compose and roughly 3 GB of free disk/RAM for the initial image build and model download.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the transmission-line image viewer, or [http://localhost:8000/inference](http://localhost:8000/inference) for object detection, CNN classification, and ANN inference.

## Database explorer

The standard CPU, NVIDIA GPU, and Jetson stacks start mongo-express automatically
after MongoDB becomes healthy. Open [http://localhost:8081](http://localhost:8081)
on the deployment host, or `http://<host-or-jetson-ip>:8081` from another computer
on the local network. The default web login is `tnb-trainer` / `password`.

To override the login, set `MONGO_EXPRESS_USERNAME` and `MONGO_EXPRESS_PASSWORD`
in the project's `.env` file, then rerun your deployment's Compose command.
These credentials protect the explorer web interface; they do not change MongoDB
authentication. The explorer allows creating, editing, and deleting data.

Open the `tnb_inference` database to inspect imported dataset records. MongoDB
persists in the `mongo-data` volume. Imported images and other dataset files
remain under `./data` (including `data/viewer` and `data/coco`); mongo-express
does not browse those files. Dataset collections appear after data is imported.

## Transmission-line image viewer

The landing-page viewer uses the supplied public Backblaze archive:

```text
https://f005.backblazeb2.com/file/jetson-assets/dataset/transmission%20line%20obj%20det.zip
```

Select **Load transmission-line dataset** in the viewer to queue a download and import. The importer runs independently from inference workers, shows progress in the page, stores its files under `data/viewer`, and does not touch legacy datasets under `data/coco`.

The archive is Pascal VOC style: every annotated image has an XML file in an `Annotations/` directory and its image in the sibling `JPEGImages/` directory. XML labels are retained exactly as authored. Invalid image/XML pairs are skipped and reported after import.

### Use your own VOC dataset

Start the stack, then upload a local Pascal VOC ZIP to activate it in the viewer:

```bash
python scripts/upload_voc_dataset.py /path/to/my-dataset.zip
```

For a non-default local address, supply `--api-url`, for example:

```bash
python scripts/upload_voc_dataset.py ./inspection-data.zip --api-url http://localhost:8000
```

The script uploads to the running local API, queues the import, waits for completion, and only switches the active viewer dataset after a successful import. A ZIP must contain matching `Annotations/*.xml` and `JPEGImages/*` paths; archive paths are validated before extraction.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /viewer/status` | Active viewer dataset, import state, categories, and download progress. |
| `POST /viewer/import-jobs` | Queue the default public transmission-line dataset import. |
| `POST /viewer/datasets` | Upload and queue a custom Pascal VOC ZIP. |
| `GET /viewer/images` | Cursor-paginated active images; supports `category_id`, `cursor`, and `limit`. |
| `GET /viewer/images/{image_id}` | Image metadata and Pascal VOC annotations. |
| `GET /viewer/images/{image_id}/file` | Stream the imported image file. |
| `POST /jobs` | Multipart field `image`; queues object detection. |
| `POST /classification/jobs` | Multipart image; queues ResNet-50 classification. |
| `GET /jobs/{job_id}` | Returns job state and result when finished. |
| `GET /ann/models` | Default ANN, registered model metadata/examples, and configuration errors. |
| `POST /ann/models/{model_id}/jobs` | Queue numeric ANN inference with JSON `{"features": {...}}`. |
| `POST /ann/models/{model_id}/image-jobs` | Queue image ANN inference with multipart field `image`. |
| `GET /health` | Checks Redis and that an inference worker is available. |

## ANN inference

Open **AI Inference → ANN Inference**. PIMA Predict is included: select **Load
example**, then **Run prediction**. Training takes place in the notebooks, outside
the serving app. Prediction requests use the existing Redis/RQ inference worker.

The default bundle in `models/pima/` contains two files:

- `model.pt`: trained PyTorch weights.
- `model.json`: architecture, ordered input names, fitted scaler, labels,
  output interpretation, example input, and training/evaluation metadata.

The included model follows the PIMA reference notebook (seed 42, stratified
80/20 split, 100 epochs). Its measured held-out accuracy is **74.03%** on 154
samples. It is an educational dataset classifier. These model files need no
training or dataset download when the application starts. The separate public
vision workflows may still download their pretrained weights on first startup.

### Deploy your notebook model

1. Complete the notebook exercises, train and evaluate, then run **Export for deployment**.
2. Copy **both** generated files into `models/<model_id>/`, keeping the folder
   name equal to `model_id` in the generated JSON. For example, MNIST exports to
   `deployment/mnist/`; copy its two files to `models/mnist/` in this repository.
3. Register the folder in `config/ann.json`:

   ```json
   {
     "default_model": "pima",
     "models": ["pima", "mnist"]
   }
   ```

4. Restart both services and reload the page:

   ```bash
   docker compose restart api worker
   ```

Use the same Compose files or Jetson wrapper as your original startup when
restarting a GPU/Jetson deployment. For an existing installation upgrading from
the training workflow, first rebuild using your normal startup command with
`--build --remove-orphans`; this removes the old training-worker container.
Existing files under `data/tabular/` are preserved but are no longer served.

Only registered models appear in the selector. PIMA and Iris-style models use
numeric fields; MNIST uses image upload and shows the exact processed grayscale
image after prediction. **Load example** uses the sample stored in the bundle.
The Iris reference notebook exports its four raw `F1`–`F4` features without
scaling and preserves its `LabelEncoder` class order. PIMA exports its fitted
scaler; each bundle uses the preprocessing from its own training notebook.

Always replace both files together after retraining. Registration and metadata
are read at startup; networks load once per worker process. If a bundle is
missing or invalid, the page reports what to fix. An invalid default never
silently switches to another model. Correct the files and restart both services.

The runtime supports the notebooks' sequential Linear/ReLU/Flatten networks
(up to 10 million parameters), float32 numeric input with no scaling or saved
standard scaling, and single-channel images with bilinear resizing, optional
inversion, and pixel scaling. Binary logits use sigmoid and the exported
threshold; multiclass logits use softmax; scalar regression returns a number.
MNIST expects a light digit on a dark background, matching its training images.

Compose mounts `config/` and `models/` read-only into the API and worker. Standalone
Docker images also include the default files. To change their locations outside
Compose, set `ANN_CONFIG` and `ANN_MODEL_DIR` consistently for both processes.
The old `/tabular/*` training endpoints have been removed.

### Reproduce the default PIMA bundle (optional)

Use a separate Python environment with CPU PyTorch installed:

```bash
python -m venv .tmp/pima-generation
.tmp/pima-generation/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.tmp/pima-generation/bin/pip install -r requirements-generation.txt
.tmp/pima-generation/bin/python scripts/generate_pima_bundle.py
```

The script supplies the intended exercise answers in memory, executes the PIMA
notebook headlessly, and uses its export cell without editing the student
notebook. It downloads the notebook's CSV and checks its recorded SHA-256;
`--dataset /path/to/diabetes.csv` uses a local copy instead. `--output-dir` chooses
an alternate destination. Generation dependencies are not installed in serving
containers. Exact training results may differ across PyTorch versions/devices;
the included JSON records the actual version, device, and measured accuracy.

## NVIDIA GPU mode

The normal command is CPU-only. For an NVIDIA GPU, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

## NVIDIA Jetson deployment

Use the Jetson-specific stack when building directly on an ARM64 Jetson. It
selects the NVIDIA-compatible base image and the exact PyTorch/Torchvision wheel pair from
the host's `/etc/nv_tegra_release`; it never uses the desktop CUDA wheels.

Install Docker, Docker Compose, and NVIDIA Container Toolkit on the Jetson,
then run from this repository:

```bash
./scripts/jetson-compose.sh up --build
```

The wrapper directs Compose build metadata away from a potentially small
`/tmp` tmpfs and into `.tmp/jetson-compose`. If the repository is itself on a
space-constrained filesystem, set `JETSON_TMPDIR` to a writable directory on a
larger persistent mount:

```bash
JETSON_TMPDIR=/mnt/ssd/compose-tmp ./scripts/jetson-compose.sh up --build
```

Only the inference worker uses the NVIDIA runtime; the API and import services
do not reserve GPU access.

The command starts the full stack and exposes the application at port 8000.
The first build downloads PyTorch and the first inference request downloads the
public model weights, so allow several GB of free storage and a stable network
connection. Application data remains in `./data` and downloaded model weights
are kept in the `torch-model-cache` volume.

Supported native builds follow the Jetson launcher wheel matrix:

| JetPack / L4T | PyTorch stack |
| --- | --- |
| 6.0 / 36.2 or 36.3 | Torch 2.3.0 + Torchvision 0.18.0a0 |
| 6.2 and patch releases / 36.4.3, 36.4.4, 36.5.0, or 36.5.2 | Torch 2.8.0 + Torchvision 0.23.0 |
| 7.2 and patch releases / 39.2.1 | Torch 2.14.1 + Torchvision 0.29.1 |

JetPack 5 and JetPack 6.1 fail before the image build: their launcher wheel
sets do not provide the Python/runtime or verified Torchvision compatibility
required by this project. For JetPack 6 wheels, the installer tries the managed
`jetson-assets` mirror first and then the launcher-recorded upstream URL.

After the stack is healthy, verify CUDA inside the GPU worker:

```bash
./scripts/jetson-compose.sh exec worker python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

For diagnostics without invoking Docker, point the resolver at the actual
release file (or a copied one) and print the selected configuration:

```bash
./scripts/jetson-compose.sh --print-env
```

## Development tests

```bash
pip install -r requirements-dev.txt
pytest
```
