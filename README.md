# TNB Inference

TNB Inference is a small asynchronous PyTorch demo with a Pascal VOC transmission-line image viewer, image classification, object detection, and numeric tabular-model training.

```text
browser → FastAPI → Redis/RQ queue → worker → database/files → browser
```

## Run it

Prerequisites: Docker Compose and roughly 3 GB of free disk/RAM for the initial image build and model download.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the transmission-line image viewer, or [http://localhost:8000/inference](http://localhost:8000/inference) for inference and tabular workflows.

## Transmission-line image viewer

The landing-page viewer uses the supplied public Backblaze archive:

```text
https://f005.backblazeb2.com/file/jetson-assets/dataset/transmission%20line%20obj%20det.zip
```

Select **Load transmission-line dataset** in the viewer to queue a download and import. The importer runs independently from inference/training workers, shows progress in the page, stores its files under `data/viewer`, and does not touch legacy datasets under `data/coco`.

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
| `POST /tabular/datasets` | Upload a CSV/XLSX/XLS file and return its preview. |
| `POST /tabular/training-jobs` | Queue ANN training from an uploaded table. |
| `GET /health` | Checks Redis and that an inference worker is available. |

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

Only the inference and training workers use the NVIDIA runtime; the API and
import services do not reserve GPU access.

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
