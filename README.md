# 🌡️ DroneTracker Thermal

Thermal-imagery drone detection & tracking with **YOLO (Ultralytics)** — trained on a custom thermal dataset and tuned specifically for small, dense, multi-drone scenes. Adds **ByteTrack** for video tracking and **SAHI** (Slicing Aided Hyper Inference) so small thermal drone signatures aren't missed at native resolution.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CingizEsgerov05/DroneTracker_Thermal/blob/main/Thermal_Training.ipynb)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Model](https://img.shields.io/badge/model-YOLO%20(Ultralytics)-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Demo

<!-- Replace with an actual GIF or screenshot: assets/demo.gif -->
`Thermal_Demo.gif`
`Thermal_Demo_MultiDrone`

## Workflow

Unlike a single run-all pipeline, this notebook is an **iterative training log** — sections are run over multiple sessions as the model improves, with checkpoints persisted to Google Drive so training survives Colab disconnects.

| Stage | What it does |
|-------|---------------|
| **Setup** | Installs `ultralytics` and `sahi` |
| **Dataset audit** | Walks the thermal dataset (train/val) and reports image dimensions — the set is a mix of `512×640` and `512×512` frames |
| **Training** | Trains YOLO on the thermal dataset (`imgsz=640`, up to 100 epochs), checkpointing every 5 epochs to Drive; resumable at any time via `model.train(resume=True)` |
| **Detection & tracking** | Runs `model.predict` and `model.track` (ByteTrack) on thermal test videos to detect and follow drones frame-to-frame |
| **Dense-scene tuning** | Inference re-tuned for small/many targets: higher resolution (`imgsz=1280`), low confidence threshold, high `max_det`, class-agnostic NMS |
| **SAHI sliced inference** | Slices each frame (sized to match the source resolution) with overlap and merges detections across slice borders (NMM) — catches small thermal drone signatures that a single full-frame pass misses |
| **Crowd fine-tuning** | A second fine-tuning pass (`copy_paste` + `mixup` augmentation) specifically teaches the model dense, multi-drone "crowd" scenes |

## Quick start

1. Open in Colab (badge above) and mount your Google Drive.
2. Update the dataset/checkpoint paths in the training and inference cells to point at your own Drive structure (paths are currently hardcoded to the author's Drive layout).
3. Run **Setup** → **Dataset audit** (optional sanity check) → **Training**.
4. For inference, load a checkpoint (`best.pt` or `last.pt`) and run the **Detection & tracking** or **SAHI** cells on your own thermal video/images.

## Tech stack

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — detection & training
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics' built-in tracker) — multi-object tracking
- [SAHI](https://github.com/obss/sahi) — sliced inference for small/dense objects
- OpenCV, PIL

## Notes & known limitations

- Dataset paths point to the author's personal Google Drive (`/content/drive/MyDrive/CV_datasets/...`) — swap these for your own before running.
- The thermal dataset itself isn't included in the repo (too large / private); bring your own in the same folder layout (`train/`, `val/`, `data.yaml`).
- This is a working training log, not a polished single-purpose script — some cells are empty or exploratory (kept for transparency into the actual training process).
- SAHI slice size is tuned to a `512×640` source frame; re-tune `slice_height`/`slice_width` for other resolutions.

## Repository structure

```
.
├── Thermal_Training.ipynb                     # Training + inference/tracking notebook
├── Thermal_Demo + Thermal_Demo_MultiDrone     # Demo test GIF / sample frames
├── README.md
└── LICENSE
```

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SAHI](https://github.com/obss/sahi) — Akyon et al., *"Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"*
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Zhang et al.

## License

MIT — see [LICENSE](LICENSE).
