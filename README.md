# Musify — Music Genre Classifier

**Find the style of your music.** Musify is a local audio-classification app. Its single application model combines frozen YAMNet and PANNs representations with a fitted 12-label classifier. Upload WAV, MP3, FLAC, OGG, M4A, or AU audio to see a predicted genre, score distribution, playback, waveform, and mel-spectrogram.

## CODTECH Internship Details

- **Intern ID:** CITS9608  
- **Full Name:** Apeksha Mahadev Mundey 
- **No. of Weeks:** 4
- **Project Name:** Musify — Music Genre Classifier
- **Project Scope:** Build and evaluate a four-source music genre classifier with track-level data splits, then provide a FastAPI inference service and React interface for audio uploads and prediction visualizations.

## The model

Musify combines two frozen pretrained audio representations:

- YAMNet frame embeddings pooled by mean, standard deviation, and maximum (3,072 values).
- A 2,048-value PANNs CNN14 embedding extracted at 16 kHz.

A weighted StandardScaler and logistic-regression classifier use the combined 5,120-value representation to predict one of 12 labels: **Western classical, Hindustani classical, Carnatic classical, blues, country, disco, hiphop, jazz, metal, pop, reggae,** or **rock**. The feature extractors are frozen; the fusion classifier is fitted from the documented training sources. Training weights balance genres and sources that contribute the same genre.

The trained classifier, preprocessing configuration, split manifest, evaluation report, and confusion matrix are in `artifacts/musify_four_dataset_hybrid_v2/`. The YAMNet and PANNs feature-extractor files must be obtained separately and are not bundled in Git.

## Training sources and recorded evaluation

The classifier was fitted using audio from GTZAN, FMA-Small, Saraga Hindustani, and Saraga Carnatic. These results describe the saved experiment and are not a guarantee for arbitrary uploads.

| Held-out source | Tracks | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| GTZAN | 148 | 81.76% | 0.8102 |
| FMA-Small | 450 | 75.11% | 0.7641 |
| Saraga Hindustani | 25 | 100.00% | 1.0000 |
| Saraga Carnatic | 56 | 100.00% | 1.0000 |
| **Pooled total** | **679** | **79.53%** | **0.8238** |

FMA-Small accounts for 450 of the 679 test tracks, so pooled accuracy is influenced most by that source. The report also includes an equal-source mean (89.22% accuracy, 0.8936 macro-F1); this is a different summary and must not be described as pooled accuracy.

Interpret the results cautiously. GTZAN, FMA-Small, and Hindustani Saraga use test splits examined in earlier experiments, so these are not a fresh blind estimate. The Carnatic split holds out Saraga release groups, but it is still the same collection. Both Indian-classical labels are represented only by Saraga, and their test sets are small. The model has no Indian-film or soundtrack label; it will assign such recordings to the closest available class. Its score distribution is not calibrated confidence.

## Setup

Use **Python 3.10–3.13**. TensorFlow 2.20 is pinned; Python 3.14 is outside the declared project range. The app downloads and caches the two feature extractors it needs on the first prediction if they are not already available locally.

From the repository root in PowerShell:

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[hybrid]"
cd frontend
npm install
cd ..
~~~

The `hybrid` extra installs TensorFlow Hub, PyTorch, torchlibrosa, and a compatible `setuptools` version. TensorFlow Hub 0.16.1 imports `pkg_resources`, so the extra constrains setuptools below 81. For development checks only, install `.[dev,hybrid]` instead. If a CPU-only PyTorch build is needed on Windows, use PyTorch's official installation selector.

Verify the environment after installation:

~~~powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import tensorflow as tf, tensorflow_hub as hub; print('TensorFlow', tf.__version__, '| TensorFlow Hub', hub.__version__)"
~~~

### Automatic feature-extractor setup

The trained fusion classifier is included. On the first prediction, Musify automatically obtains only the two feature extractors used by this application: PANNs CNN14 and YAMNet. PANNs is downloaded to the ignored local path `artifacts/pretrained/panns/` and verified against its published MD5 checksum. YAMNet is fetched from the versioned TensorFlow Hub handle and cached by TensorFlow Hub (set `TFHUB_CACHE_DIR` to choose the cache location). Later predictions reuse the cached files.

The PANNs checkpoint is about 343 MiB, so the first prediction can take a while and needs internet access and sufficient disk space. Musify does not download the training datasets or the other experimental model variants to run inference.

To download PANNs before starting the app, use the optional helper:

~~~powershell
.\scripts\download_panns_checkpoint.ps1
~~~

To use a local YAMNet copy instead of the TensorFlow Hub cache, download and extract it manually:

~~~powershell
New-Item -ItemType Directory -Force .\artifacts\pretrained\yamnet | Out-Null
curl.exe -L --fail -o .\artifacts\pretrained\yamnet.tar.gz "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
tar -xzf .\artifacts\pretrained\yamnet.tar.gz -C .\artifacts\pretrained\yamnet
~~~

The expected local paths are `artifacts/pretrained/yamnet/` and `artifacts/pretrained/panns/Cnn14_16k_mAP_0.438.pth`. These downloaded files are ignored by Git. Review the upstream terms before downloading or redistributing them.

## Run the application

Start the API from the repository root in one PowerShell terminal:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn genre_classifier.api:app --app-dir src --host 127.0.0.1 --port 8000
~~~

Start the frontend in another terminal:

~~~powershell
cd frontend
npm run dev -- --host 127.0.0.1
~~~

Open **http://127.0.0.1:5173/** for Musify. The API is at **http://127.0.0.1:8000/**; its root and `/health` show service status, and `/docs` provides the API schema. The application exposes only the four-source model. Its artifacts are selected by default in the frontend and configured by `FOUR_SOURCE_MODEL_DIR` in the API.

## Reproduce training

Training from scratch requires locally obtained source audio or prepared feature caches for the documented sources, plus YAMNet/PANNs files (downloaded automatically when missing). Dataset audio and feature caches are not included in this repository. Obtain each source under its applicable terms. For the Kaggle GTZAN archive used during development, the audio folder was `data/raw/Data/genres_original`.

The four-source trainer retains the GTZAN/FMA-Small/Hindustani assignments, removes exact Saraga duplicates, splits Saraga Carnatic by release group before windowing, and uses all 30-second Saraga windows at a 15-second hop. The final saved run used seed `20261022`, `C=0.001`, and the fused embedding option. From the repository root, with inputs/caches prepared at the paths expected by the trainer:

~~~powershell
.\.venv\Scripts\python.exe -m genre_classifier.train_four_dataset_hybrid --hindustani-archive data/raw/saraga_hindustani/saraga1.5_hindustani.zip --carnatic-archive data/raw/saraga_carnatic/saraga1.5_carnatic.repaired_verified.zip --all-windows --feature-sets hybrid --output artifacts/musify_four_dataset_hybrid_reproduction --work-dir artifacts/work_musify_four_dataset_hybrid_reproduction
~~~

The new output directory keeps a reproduction run separate from the included submission artifact. Training writes the classifier, preprocessing config, split manifest, evaluation JSON, and confusion matrix.

## Repository contents

- `artifacts/musify_four_dataset_hybrid_v2/` — submission classifier and its evaluation files.
- `src/genre_classifier/` — audio preprocessing, YAMNet/PANNs feature extraction, inference API, and training code.
- `frontend/` — the single-model React/Vite interface.
- `scripts/` — checksum-verifying PANNs checkpoint downloader.
- `tests/` — lightweight audio and track-splitting checks.
- `screenshots/` — application screenshot; the model evaluation output is included with the saved artifacts.
- `THIRD_PARTY_LICENSES.md` — third-party attributions and dataset/model term notes.

Raw audio, feature caches, and downloaded pretrained checkpoints are excluded from Git. Historical experiment source files may remain in the repository for development reference; the frontend and API serve only the four-source model, and only its trained artifact is included as the submission model. Missing feature extractors are fetched on first inference and cached locally.

## Dataset and model terms

Third-party audio, pretrained extractors, and software packages have separate terms. Source audio and pretrained feature-extractor weights are not included in Git, but the fitted classifier is included. This project has not determined whether the trained artifact is an adapted work under its training sources' terms. Review `THIRD_PARTY_LICENSES.md` and `artifacts/musify_four_dataset_hybrid_v2/MODEL_NOTICE.md`, preserve required attribution, and check applicable terms before redistribution or commercial use. The root MIT license covers Musify-authored code only; it does not grant rights to the fitted classifier or third-party materials. These notes are not legal advice or a license grant.

## Tests

Run the lightweight project checks without downloading the datasets or loading the pretrained extractors:

~~~powershell
.\.venv\Scripts\python.exe -m pytest
~~~

Build the frontend production bundle:

~~~powershell
cd frontend
npm run build
~~~
