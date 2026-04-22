# Gait Tracking Live Streaming Extension

This repository is the **live-streaming extension** of our offline gait-tracking course project.

It is built on top of:
- the original upstream **All-in-One-Gait** project by the OpenGait team
- the upstream **OpenGait** framework

Please also see:
- offline/stable project repo: [PritomKumarPaul/gait_tracking](https://github.com/PritomKumarPaul/gait_tracking)
- upstream OpenGait releases: [https://github.com/ShiqiYu/OpenGait/releases](https://github.com/ShiqiYu/OpenGait/releases)

This extension keeps the original tracking, segmentation, and gait-recognition logic, then adds:
- offline gallery-probe demos for closed-set testing
- cached gallery workflows for repeated experiments
- buffered realtime-feasibility testing
- a final FastAPI-based live webcam demo

## Upstream Credit

This work depends heavily on the original ideas, code, and checkpoints from:
- **All-in-One-Gait**
- **OpenGait**
- **ByteTrack**
- **PaddleSeg**

Please keep the original upstream attribution intact when using or presenting this project.

Useful local reference:
- upstream README copy: `README_ORIGINAL_UPSTREAM.md`
- offline project README: `README.md`

## What This Repository Contains

The main addition in this repo is the **live streaming branch** of the project.

Important folders:
- `OpenGait/tools/`
  New and modified analysis scripts built around the OpenGait pipeline.
- `clean_demo_v2/`
  Stable offline closed-set gallery-probe demo.
- `live_demo/`
  Buffered realtime testing, gallery-cache tools, and the final live webcam app.

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

```bash
cd /home/ppaul11/All-in-One-Gait
conda create -y -n allinonegait python=3.8
conda activate allinonegait
pip install -r requirements.txt
pip install yolox
pip install gradio fastapi uvicorn pyngrok aiortc
```

Additional notes:
- `gradio` is used for the stable offline closed-set demo UI.
- `fastapi` and `uvicorn` are used for the final live webcam web app.
- `pyngrok` is used only if you want to create a public tunnel for the FastAPI app.
- `aiortc` was installed during live-streaming experimentation and transport work.

If needed, verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

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
cd /home/ppaul11/All-in-One-Gait/OpenGait/demo/checkpoints/bytetrack_model
pip install --upgrade --no-cache-dir gdown
gdown https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5
```

### 2. Segmentation Model

Expected extracted directory:

```bash
OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax
```

Download command:

```bash
cd /home/ppaul11/All-in-One-Gait/OpenGait/demo/checkpoints
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
cd /home/ppaul11/All-in-One-Gait/OpenGait/demo/checkpoints
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
cd /home/ppaul11/All-in-One-Gait/OpenGait/demo/checkpoints/gait_model
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
cd /home/ppaul11/All-in-One-Gait
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

## Gallery Cache Building for Live Demo

Before the live webcam demo, gallery caches must be built from the selected gallery videos.

For GaitBase:

```bash
cd /home/ppaul11/All-in-One-Gait
conda activate allinonegait

python live_demo/build_pritom_coco_gallery.py \
  --model grew_gaitbase \
  --pritom-gallery-video /home/ppaul11/All-in-One-Gait/live_demo/gallery/pritom.mp4 \
  --coco-gallery-video /home/ppaul11/All-in-One-Gait/live_demo/gallery/coco.mp4 \
  --force
```

For GaitGL:

```bash
python live_demo/build_pritom_coco_gallery.py \
  --model grew_gaitgl \
  --pritom-gallery-video /home/ppaul11/All-in-One-Gait/live_demo/gallery/pritom.mp4 \
  --coco-gallery-video /home/ppaul11/All-in-One-Gait/live_demo/gallery/coco.mp4 \
  --force
```

Verify caches:

```bash
sed -n '1,120p' live_demo/cache/pritom_coco_gallery.json
sed -n '1,120p' live_demo/cache/pritom_coco_gallery_grew_gaitgl.json
```

## Buffered Realtime Feasibility Testing

This was used to see whether the original pipeline could be pushed close to realtime before building the final live app.

Run:

```bash
cd /home/ppaul11/All-in-One-Gait
conda activate allinonegait
GPU_ID=1 MODEL=grew_gaitbase PROBE=live_demo/prepared_inputs/test1probe_720p30.mp4 \
  bash live_demo/run_v3_realtime_profile.sh
```

Or directly:

```bash
CUDA_VISIBLE_DEVICES=1 python live_demo/run_buffered_live_probe.py \
  --video live_demo/prepared_inputs/test1probe_720p30.mp4 \
  --model grew_gaitbase \
  --process-every-n 5 \
  --assigned-process-every-n 10 \
  --detector-input-size 0 \
  --work-frame-max-side 0 \
  --min-detection-score 0 \
  --output-max-side 480 \
  --silhouette-every-n-processed 1 \
  --identity-buffer-frames 5 \
  --max-seconds 30 \
  --log-every 0 \
  --quiet \
  --write-output-video
```

## Final Live Webcam Demo

The final live product is the FastAPI-based live webcam demo.

This product uses:
- **FastAPI** for the backend application
- **Uvicorn** as the application server
- optional **pyngrok/ngrok** for public URL exposure

Install requirement reminder:

```bash
conda activate allinonegait
pip install fastapi uvicorn pyngrok
```

Run locally on the server:

```bash
cd /home/ppaul11/All-in-One-Gait
conda activate allinonegait
LIVE_DEMO_PUBLIC=0 LIVE_DEMO_GPU_ID=1 LIVE_DEMO_PORT=8010 python live_demo/fastapi_live_webcam_app.py
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
LIVE_DEMO_PUBLIC=1 LIVE_DEMO_GPU_ID=1 LIVE_DEMO_PORT=8010 python live_demo/fastapi_live_webcam_app.py
```

If ngrok is not configured, the app will still run locally on the server but no public URL will be created.

## Notes on What Worked and What Did Not

### Worked
- offline closed-set gallery-probe demo
- gallery cache building
- buffered realtime-feasibility testing
- final FastAPI-based live webcam demo

### Did Not Become Final Product
- probe-only unknown matching with threshold and margin
  Reason: not stable enough for the final deployment story.
- direct Gradio browser-webcam streaming as the main live backend
  Reason: browser preview and backend frame delivery were unreliable in this environment.

## Attribution Reminder

This repository is an academic derivative project. Please keep attribution to the upstream OpenGait / All-in-One-Gait / ByteTrack / PaddleSeg work in any report, README, or presentation.
