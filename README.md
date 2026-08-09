# IFCB Inference App

A [web application](https://irfcb-classify.hf.space/) for running inference on phytoplankton images from Imaging FlowCytobot (IFCB) instruments using a fine-tuned ResNet-50 model for the Skagerrak, Kattegat, and Baltic sea. Built with [Gradio](https://www.gradio.app/).

## Features

- Single image classification with top-5 predictions and confidence bars
- Batch classification via ZIP upload (up to 10,000 images)
- Paginated image gallery with sorting by name or dimension
- Per-class F1 optimised thresholds displayed on prediction bars, and the resolved class they imply
- Auto-discovery of models from the `data/models/` directory
- API endpoints for programmatic access (`predict_scores`, `get_thresholds`)
- Scores match [ifcb-pytorch-classify](https://github.com/nodc-sweden/ifcb-pytorch-classify) — see below

## Models

The app auto-discovers models from the `data/models/` directory. Each model needs a subdirectory containing `weights.pth`/`weights.pt`, and optionally `thresholds.json`, `classes.txt` (if classes are not stored together weights and thresholds) and `about.md`. Model weights are not included in the GitHub repository — contact the author to obtain them.

A checkpoint exported by ifcb-classify describes itself: the architecture, input
size, transform and dataset statistics are read from the training config it
carries, and the About panel reports what was used. Nothing needs configuring per
model. Older bare state-dicts carry none of that and are scored as ResNet-50 at
224 x 224 with `dataset_squarepad`, which is what the pipeline assumes for them
too.

The included example configuration (SMHI-NIVA-ResNet50-V5) is:

- **Architecture:** ResNet-50
- **Input:** 224 x 224 px (square-padded with adaptive background colour)
- **Classes:** 109
- **Training data:** [SMHI IFCB Plankton Image Reference Library](https://doi.org/10.17044/scilifelab.25883455) and images provided by the Norwegian Institute for Water Research (NIVA)

### Training your own model

This app is inference-only. To train your own PyTorch classification model for IFCB data, see [IFCBClassify_DEMO](https://github.com/EuropeanIFCBGroup/IFCBClassify_DEMO) or [ifcb-pytorch-classify](https://github.com/nodc-sweden/ifcb-pytorch-classify). Once trained, place the exported `weights.pth`/`weights.pt` and an optional classlist (`classes.txt`) in a new subdirectory under `data/models/` and the app will pick it up automatically.

## Requirements

- Python 3.11+
- PyTorch (CPU-only by default via `requirements.txt`)

### GPU Support

The default `requirements.txt` installs CPU-only PyTorch. The app automatically detects and uses CUDA or MPS (Apple Silicon) when available. To enable GPU acceleration, reinstall PyTorch with CUDA support:

```bash
pip install -r requirements.txt

# Then reinstall PyTorch with CUDA (check https://pytorch.org/get-started for the latest commands)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # Use your CUDA version
```

## Local Setup

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

pip install -r requirements.txt
python main.py
```

The app starts at `http://localhost:7860`.

## Docker

```bash
docker build -t ifcb-classification-app .
docker run -p 7860:7860 ifcb-classification-app
```

## Agreement with the ifcb-classify pipeline

This app and the [ifcb-pytorch-classify](https://github.com/nodc-sweden/ifcb-pytorch-classify)
`infer` command are meant to give the same answer for the same image, so a result
checked here can be trusted to match a bin classified in bulk. Two things make
that hold:

- **Nothing about the model is hardcoded.** The architecture, input size,
  transform and dataset statistics all come from the training config the
  checkpoint carries, so a model trained with a different backbone, a different
  input size or a normalised transform scores correctly without changing this
  app. Checkpoints that carry no config (a bare state-dict plus `classes.txt`)
  fall back to the same defaults the pipeline assumes: ResNet-50, 224 x 224,
  `dataset_squarepad`.
- **The training augmentation never runs.** A transform name bundles
  preprocessing (padding, resize, normalisation) with augmentation (random flips
  and brightness/contrast jitter). Scoring through the augmented pipeline makes
  every result a single random draw, so the app resolves the checkpoint's name to
  its augmentation-free counterpart, exactly as the pipeline's
  `eval_transform_name` does.

`tests/test_pipeline_equivalence.py` asserts this against the real pipeline on
real IFCB ROIs. It needs a checkout of ifcb-classify and a directory of raw bins:

```bash
pip install pytest
IFCB_CLASSIFY_SRC=/path/to/ifcb-pytorch-classify/src \
IFCB_DATA_DIR=/path/to/ifcb/data \
python -m pytest tests/
```

Without those it skips, and the rest of the suite still runs.

### Thresholds

Per-class thresholds are fitted on the validation split. A file written before
that split stopped being augmented was fitted against randomly jittered and
flipped images, so it no longer matches its model's operating point. Such a file
records no `validation_transform` key; the app reports that in the About panel
and returns the key from `get_thresholds`. Scores are unaffected — only the
accept/reject boundary moves.

A model trained by a current release of ifcb-classify needs nothing here: it
records the key itself. `scripts/recompute_thresholds.py` in the pipeline
repository is for models trained before the change, which it refits without
retraining.

Training writes `{run_name}_thresholds_and_metrics.json`, and the app looks for
that name as well as a plain `thresholds.json`, so a model directory can be
copied out of a training run as-is. If a directory holds several run-named
thresholds files the app will not guess between them and applies none — leave one,
or rename the intended file to `thresholds.json`.

## API Endpoints

The app exposes two API endpoints alongside the web UI:

- **`predict_scores`** — Classify an image and return every class score, plus the
  class the pipeline would record for it:
  `{"class_labels": [...], "scores": [...], "class_name_auto": ..., "class_name": ...}`.
  `class_name_auto` is the highest-scoring class; `class_name` is the same label
  after its threshold is applied, or `"unclassified"` when the score falls short.
- **`get_thresholds`** — Return per-class F1 thresholds, class labels, and the
  `validation_transform` the thresholds were fitted through (`null` for a file
  written before that was recorded).

See the Gradio API docs at `http://localhost:7860/?view=api` when the app is running.

## Project Structure

```
├── main.py                 # Event handlers, UI layout, app entry point
├── model.py                # Model loading, inference, prediction rendering
├── preprocessing.py        # Scoring transforms, mirroring ifcb-classify
├── architectures.py        # Rebuilding a checkpoint's backbone
├── checkpoints.py          # Reading pipeline and legacy checkpoints
├── viewer.py               # Display scaling, and what actually gets classified
├── session.py              # Session state, ZIP handling, gallery helpers
├── tests/                  # Unit tests and pipeline-equivalence tests
├── data/
│   └── models/
│       └── SMHI-NIVA-ResNet50-V5/
│           ├── weights.pth      # Model weights (git-ignored)
│           ├── classes.txt      # Class labels
│           ├── thresholds.json  # Per-class F1 thresholds
│           └── about.md         # Model description (optional)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata and version
├── Dockerfile
└── LICENSE
```

## License

[MIT](LICENSE)
