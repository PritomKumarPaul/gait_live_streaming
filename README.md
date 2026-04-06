# Gait Entry Counting Project

This repository is a course-project fork built on top of the original **All-in-One-Gait / OpenGait** codebase.

Upstream sources:
- Original project base: `All-in-One-Gait`
- Original gait framework: [OpenGait](https://github.com/ShiqiYu/OpenGait)
- Tracking base: [ByteTrack](https://github.com/ifzhang/ByteTrack)
- Segmentation base: [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

For the original upstream project description and setup notes, please refer to the original repository and OpenGait project pages linked above.

## What This Repository Adds

This fork keeps the original tracking, segmentation, and gait-recognition pipeline, then adds a course-project workflow for:
- probe-only gait identity matching without a fixed gallery
- entry counting from raw videos
- cross-video identity comparison
- threshold tuning for deciding whether two entry embeddings belong to the same person

The main idea is:
1. detect and track a person in video
2. segment the tracked person into silhouettes
3. extract gait embeddings from silhouette sequences
4. compare embeddings with cosine similarity
5. count how many entry events occurred and whether repeated entries are the same person

## Repository Structure

Important folders:
- `OpenGait/`: upstream gait framework and demo code
- `OpenGait/demo/libs/`: tracking, segmentation, and demo-model helper code
- `OpenGait/tools/`: new scripts added for this project
- `OpenGait/demo/checkpoints/`: local model weights and external assets
- `OpenGait/demo/output/`: generated tracking, silhouettes, and experiment outputs
- `OpenGait/output/`: generated JSON summaries
- `team2videos/`: local test videos used in this project

New scripts added in this fork:
- `OpenGait/tools/probe_only_entry_analysis.py`
  Purpose: run the full pipeline on one video or one directory of videos and assign identities with cosine matching.
- `OpenGait/tools/compare_two_videos.py`
  Purpose: run two videos jointly and check whether entries across the two videos are matched as the same person.
- `OpenGait/tools/evaluate_two_video_thresholds.py`
  Purpose: sweep multiple thresholds and multiple checkpoints to find a good setting.
- `OpenGait/tools/run_probe_only_parallel.py`
  Purpose: distribute videos across multiple GPUs with one worker process per GPU.
- `OpenGait/tools/evaluate_gaitbase_casiab.py`
  Purpose: local evaluation helper for the current GaitBase checkpoint on CASIA-B.

## Available Model Profiles

The current scripts support these model keys:
- `grew_gaitbase`
  Uses the GREW GaitBase checkpoint.
- `grew_gaitgl`
  Uses the GREW GaitGL checkpoint.
- `current_gaitbase`
  Uses the local pre-existing GaitBase checkpoint already present in the repo layout.

Recommended default from our experiments so far:
- model: `grew_gaitbase`
- cosine threshold: `0.97`

## Checkpoints And Large Files

Do not upload large checkpoints, generated outputs, or local videos to GitHub.

Examples that should stay out of the repository:
- `.pt` checkpoint files
- `.zip` checkpoint archives
- `OpenGait/demo/output/`
- `OpenGait/output/*.json` if they are experiment artifacts rather than curated examples
- `team2videos/`
- large datasets such as `casiab/` and `casiab-pkl/`

Required external checkpoints are expected locally under:
- `OpenGait/demo/checkpoints/bytetrack_model/`
- `OpenGait/demo/checkpoints/gait_model/`
- `OpenGait/demo/checkpoints/seg_model/`

At the time of writing, the code expects these important files:
- tracking: `OpenGait/demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar`
- segmentation: `OpenGait/demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/...`
- GREW GaitBase: `OpenGait/demo/checkpoints/gait_model/GREW/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-180000.pt`
- GREW GaitGL: `OpenGait/demo/checkpoints/gait_model/GaitGL/checkpoints/GaitGL-250000.pt`

## Setup

Example environment:
```bash
conda create -y -n allinonegait python=3.8
conda activate allinonegait
pip install -r requirements.txt
pip install yolox
```

If your machine uses CUDA, verify it with:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Download Required Assets

### 1. Tracking checkpoint

Put this file in:
`OpenGait/demo/checkpoints/bytetrack_model/`

Expected file:
`bytetrack_x_mot17.pth.tar`

### 2. Segmentation checkpoint

Put the PaddleSeg inference model in:
`OpenGait/demo/checkpoints/seg_model/`

Expected extracted directory:
`human_pp_humansegv2_mobile_192x192_inference_model_with_softmax`

### 3. Gait checkpoints

GREW GaitBase release asset:
`pretrained_grew_gaitbase.zip`

GREW GaitGL release asset:
`pretrained_grew_gaitgl.zip`

Extract them into:
`OpenGait/demo/checkpoints/gait_model/`

## Main Workflows

### 1. Analyze a single video

```bash
python OpenGait/tools/probe_only_entry_analysis.py \
  --video-path /abs/path/to/video.mp4 \
  --model grew_gaitbase \
  --cosine-threshold 0.97 \
  --work-root /abs/path/to/workdir \
  --output-json /abs/path/to/result.json
```

### 2. Compare two videos jointly

```bash
python OpenGait/tools/compare_two_videos.py \
  --video-a /abs/path/to/video_a.mp4 \
  --video-b /abs/path/to/video_b.mp4 \
  --model grew_gaitbase \
  --cosine-threshold 0.97 \
  --work-root /abs/path/to/workdir \
  --output-json /abs/path/to/result.json
```

### 3. Sweep thresholds and checkpoints

```bash
python OpenGait/tools/evaluate_two_video_thresholds.py \
  --video-a /abs/path/to/video_a.mp4 \
  --video-b /abs/path/to/video_b.mp4 \
  --models grew_gaitbase,grew_gaitgl \
  --thresholds 0.75,0.90,0.95,0.97,0.975,0.98,0.985 \
  --work-root /abs/path/to/workdir \
  --output-json /abs/path/to/result.json
```

### 4. Run many videos across multiple GPUs

```bash
python OpenGait/tools/run_probe_only_parallel.py \
  --video-dir /abs/path/to/video_folder \
  --model grew_gaitbase \
  --gpu-ids 0,1,2,3 \
  --poll-seconds 5
```

## What The Outputs Mean

The JSON outputs typically contain:
- `summary`
  Counts of processed entries, skipped entries, and estimated unique people.
- `people`
  Grouped identities and which entry keys were assigned to them.
- `entries`
  Per-entry information such as track id, frame count, assigned person id, and best cosine similarity.
- `config`
  Which checkpoint and threshold were used.

Interpretation:
- low threshold: more aggressive merging across entries
- high threshold: more aggressive splitting into separate identities

## What To Upload To GitHub

Recommended:
- source code
- configs
- lightweight documentation
- small example JSONs only if you intentionally want them as examples

Recommended not to upload:
- checkpoints
- downloaded archives
- generated videos
- generated silhouettes
- raw local test videos
- notebooks with private or messy experiment state unless you really want to publish them

## Suggested Pre-Push Checklist

Before pushing:
1. remove or ignore `.pt`, `.zip`, generated output folders, and local datasets
2. make sure the new `README.md` is the one at repo root
3. keep any local-only reference files out of the GitHub repository
4. verify paths in the scripts still match the intended repository layout
5. make sure any course-private videos are not committed

## Attribution

This repository is a derivative academic project and depends heavily on the original code and ideas from:
- OpenGait and the Shiqi Yu Group
- ByteTrack
- PaddleSeg

If you use this fork, please also read and cite the upstream work where appropriate.
