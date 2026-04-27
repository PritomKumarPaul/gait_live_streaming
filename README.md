# Gait Tracking Live Streaming Extension

This repository is the **live-streaming extension** of our offline gait-tracking course project.

It is built on top of:
- the original upstream **All-in-One-Gait** project by the OpenGait team
- the upstream **OpenGait** framework

Please also see:
- upstream OpenGait releases: [https://github.com/ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)

This extension keeps the original tracking, segmentation, and gait-recognition logic, then adds:
- offline gallery-probe demos for closed-set testing
- a stable offline Gradio demo in `clean_demo_v2`
- a final generic FastAPI-based live webcam demo in `live_demo_clean`
- older live experiment folders kept for reference

## Upstream Credit

This work depends heavily on the original ideas, code, and checkpoints from:
- **All-in-One-Gait**
- **OpenGait**
- **ByteTrack**
- **PaddleSeg**

Please keep the original upstream attribution intact when using or presenting this project.

## What This Repository Contains

The main addition in this repo is the **live streaming branch** of the project.

Important folders:
- `OpenGait/tools/`
  New and modified analysis scripts built around the OpenGait pipeline.
- `clean_demo_v2/`
  Stable offline closed-set gallery-probe demo.
- `live_demo/`
  Older buffered realtime testing and intermediate live-demo experiments.
- `live_demo_clean/`
  Final generic live webcam streaming demo for any number of gallery identities.

## What Should Not Be Uploaded

This repository should **not** contain:
- model checkpoints (`.pt`, `.pth`, `.pth.tar`)
- downloaded checkpoint archives (`.zip`, `.tar.gz`)
- generated output videos
- generated silhouette folders
- gallery or probe videos
- live gallery caches (`.npz`, `.json`) generated from private videos
- local datasets

The `.gitignore` in this repo excludes the important output and local-data paths for this live extension.

## Environment Setup

The most reliable setup path is the validated step-by-step install below.

Notes:
- clone the repo first if you are starting from scratch
- commands below assume you are already inside the repository root
- replace `ppaul11` in any absolute path examples with your own username or home path as needed

```bash
git clone https://github.com/PritomKumarPaul/gait_live_streaming.git /home/ppaul11/All-in-One-Gait
cd /home/ppaul11/All-in-One-Gait
```

```bash
conda create -y -n allinonegait python=3.8
conda activate allinonegait
```

Install PyTorch first:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu116 \
  torch==1.12.1+cu116 \
  torchvision==0.13.1+cu116 \
  torchaudio==0.12.1+cu116
```

Install build helpers and NumPy:

```bash
pip install Cython==3.0.11 wheel setuptools
pip install numpy==1.23.5
```

Install `cython-bbox` before the rest:

```bash
pip install cython-bbox==0.1.3 --no-build-isolation
```

Install `pycocotools` explicitly:

```bash
pip install pycocotools==2.0.7
```

Install the remaining Python packages:

```bash
pip install filelock filterpy gdown h5py kornia lap loguru motmetrics ninja onnx onnxoptimizer onnxruntime opencv-python==4.5.5.64 Pillow prettytable pyyaml scikit-image scikit-learn scipy tabulate tensorboard thop tqdm "visualdl>=2.2.0" gradio fastapi uvicorn pyngrok aiortc imageio-ffmpeg paddlepaddle-gpu paddleseg
```

If `cython-bbox` fails with `gcc: No such file or directory`, install a compiler first:

```bash
sudo apt update
sudo apt install -y build-essential
```

Then retry:

```bash
pip install Cython==3.0.11 wheel setuptools
pip install numpy==1.23.5
pip install cython-bbox==0.1.3 --no-build-isolation
```

Additional notes:
- `gradio` is used for the stable offline closed-set demo UI.
- `fastapi` and `uvicorn` are used for the final live webcam web app.
- `pyngrok` is used only if you want to create a public tunnel for the FastAPI app.
- `aiortc` was installed during live-streaming experimentation and transport work.
- do not install the PyPI `yolox` package for this repo; the project uses the bundled local YOLOX code under `OpenGait/demo/libs/yolox`

If needed, verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected compatible PyTorch stack:

```bash
python -c "import torch, torchvision, torchaudio; print(torch.__version__); print(torchvision.__version__); print(torchaudio.__version__)"
```

For the project server with A100 GPUs, the expected torch build is:
- `torch==1.12.1+cu116`
- `torchvision==0.13.1+cu116`
- `torchaudio==0.12.1+cu116`

Important installation note:
- install PyTorch first
- install `Cython`, `numpy==1.23.5`, `cython-bbox`, and `pycocotools` before the rest of the packages

## Required External Weights and Assets

The code expects three major model families:
- tracking model
- segmentation model
- gait-recognition model

### 1. Tracking Model

Expected path:

```bash
OpenGait/demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar
```

Download command used in the original upstream project:

```bash
pip install --upgrade --no-cache-dir gdown
cd OpenGait/demo/checkpoints/bytetrack_model
gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5
```

### 2. Segmentation Model

Expected extracted directory:

```bash
OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax
```

Download command:

```bash
cd OpenGait/demo/checkpoints
mkdir -p seg_model
cd seg_model
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip
unzip human_pp_humansegv2_mobile_192x192_inference_model_with_softmax.zip
```

### 3. GREW GaitBase Checkpoint

From OpenGait v2.0 releases:
- asset: `pretrained_grew_gaitbase.zip`

Download and extract commands:

```bash
cd OpenGait/demo/checkpoints
mkdir -p gait_model
cd gait_model
wget https://github.com/ShiqiYu/OpenGait/releases/download/v2.0/pretrained_grew_gaitbase.zip
unzip pretrained_grew_gaitbase.zip
```

Expected important checkpoint location:

```bash
OpenGait/demo/checkpoints/gait_model/GREW/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-180000.pt
```

### 4. GREW GaitGL Checkpoint

From OpenGait v1.1 releases:
- asset: `pretrained_grew_gaitgl.zip`

Download and extract commands:

```bash
cd OpenGait/demo/checkpoints/gait_model
wget https://github.com/ShiqiYu/OpenGait/releases/download/v1.1/pretrained_grew_gaitgl.zip
unzip pretrained_grew_gaitgl.zip
```

Expected important checkpoint location:

```bash
OpenGait/demo/checkpoints/gait_model/GaitGL/checkpoints/GaitGL-250000.pt
```

## Available Model Keys

The current scripts support:
- `grew_gaitbase`
- `grew_gaitgl`
- `current_gaitbase`

Recommended default:
- `grew_gaitbase`

## Stable Offline Demo

This is the most stable closed-set product in the project.

This product uses a **Gradio** interface.

Install requirement reminder:

```bash
conda activate allinonegait
pip install gradio
```

Main launch command:

```bash
conda activate allinonegait
python clean_demo_v2/gradio_app.py
```

Or set a specific GPU:

```bash
LIVE_DEMO_GPU_ID=1 python clean_demo_v2/gradio_app.py
```

Closed-set shell wrapper:

```bash
MODEL=grew_gaitbase bash clean_demo_v2/run_pritom_coco_test1_closed.sh
```

Single-target threshold wrapper:

```bash
MODEL=grew_gaitbase THRESHOLD=0.97 bash clean_demo_v2/run_test1_threshold.sh
```

## Final Live Webcam Demo (`live_demo_clean`)

The final live product is the generic FastAPI-based live webcam demo in `live_demo_clean/`.

This product uses:
- **FastAPI** for the backend application
- **Uvicorn** as the application server
- optional **pyngrok/ngrok** for public URL exposure

Install requirement reminder:

```bash
conda activate allinonegait
pip install fastapi uvicorn pyngrok
```

### 1. Add gallery videos

Put at least 2 gallery videos into:

```bash
/home/ppaul11/All-in-One-Gait/live_demo_clean/gallery
```

Example:

```bash
mkdir -p /home/ppaul11/All-in-One-Gait/live_demo_clean/gallery
cp /path/to/personA.mp4 /home/ppaul11/All-in-One-Gait/live_demo_clean/gallery/
cp /path/to/personB.mp4 /home/ppaul11/All-in-One-Gait/live_demo_clean/gallery/
```

The gallery builder will sort the files and assign:
- first file -> `person1`
- second file -> `person2`
- third file -> `person3`

### 2. Build live gallery cache

For GaitBase:

```bash
conda activate allinonegait

python live_demo_clean/build_generic_gallery.py \
  --gallery-dir /home/ppaul11/All-in-One-Gait/live_demo_clean/gallery \
  --model grew_gaitbase \
  --force
```

For GaitGL:

```bash
python live_demo_clean/build_generic_gallery.py \
  --gallery-dir /home/ppaul11/All-in-One-Gait/live_demo_clean/gallery \
  --model grew_gaitgl \
  --force
```

Verify cache metadata:

```bash
sed -n '1,160p' live_demo_clean/cache/generic_gallery.json
sed -n '1,160p' live_demo_clean/cache/generic_gallery_grew_gaitgl.json
```

### 3. Run locally on the server

```bash
conda activate allinonegait
LIVE_DEMO_PUBLIC=0 LIVE_DEMO_GPU_ID=1 LIVE_DEMO_PORT=8011 python live_demo_clean/fastapi_live_webcam_app.py
```

This starts a server-hosted app. If your laptop can reach the server, you can open it from the laptop browser.

### Public URL Option

To expose the FastAPI app publicly, ngrok can be used.

You need:
- an ngrok account
- a verified account if required by ngrok
- an authtoken

Set the token:

```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

or export it:

```bash
export NGROK_AUTHTOKEN=YOUR_TOKEN_HERE
```

Then run:

```bash
LIVE_DEMO_PUBLIC=1 LIVE_DEMO_GPU_ID=1 LIVE_DEMO_PORT=8011 python live_demo_clean/fastapi_live_webcam_app.py
```

If ngrok is not configured, the app will still run locally on the server but no public URL will be created.

### What the clean live app supports

- any number of gallery identities
- generic labels `person1`, `person2`, ...
- gallery mapping display, for example `person1 <- coco.mp4`
- top-2 similarity score display
- configurable segmentation warmup
- configurable identity silhouette buffer
- explicit `unassigned` reporting when a track disappears before enough silhouettes are collected

## Older Live Experiments

The `live_demo/` folder is still kept in this repository as the older experimentation area.

It includes:
- buffered realtime feasibility testing
- older Gradio live attempts
- earlier fixed-gallery tools

These are useful for reference, but the maintained final live product is now `live_demo_clean/`.

## Notes on What Worked and What Did Not

### Worked
- offline closed-set gallery-probe demo
- final generic live gallery cache building
- final FastAPI-based live webcam demo in `live_demo_clean`

### Did Not Become Final Product
- probe-only unknown matching with threshold and margin
  Reason: not stable enough for the final deployment story.
- direct Gradio browser-webcam streaming as the main live backend
  Reason: browser preview and backend frame delivery were unreliable in this environment.
- older `live_demo/` realtime-feasibility experiments
  Reason: useful for testing and ablation, but replaced by the cleaner generic `live_demo_clean/` pipeline as the maintained final streaming demo.

## Attribution Reminder

This repository is an academic derivative project. Please keep attribution to the upstream OpenGait / All-in-One-Gait / ByteTrack / PaddleSeg work in any report, README, or presentation.
