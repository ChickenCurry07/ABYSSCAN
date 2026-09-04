# ABYSSCAN — Side-Scan Sonar Anomaly Detection MVP

Sonar image → preprocessing → unsupervised AI anomaly detection → highlighted regions →
confidence → priority.

A Streamlit app that flags suspicious regions in side-scan sonar imagery — marine debris,
shipwreck fragments, or unidentified seabed objects — without needing any labeled training
data. Comes with a built-in synthetic sonar image generator, so it's fully demo-able before
you have a real dataset.

## What it does

1. **Upload** a side-scan sonar image, or use the built-in synthetic one.
2. **Preprocess**: non-local means denoising (removes sonar speckle) + CLAHE contrast
   enhancement (brings out faint returns).
3. **Detect**: the image is split into blocks, each described by a small feature vector
   (mean intensity, intensity spread, peak intensity, edge energy). An **Isolation
   Forest** — an unsupervised anomaly detector — is fit on this image's own blocks (no
   labels needed) and flags the blocks that are statistically unusual.
4. **Group + score**: flagged blocks are merged into bounding-box regions, each with an
   anomaly confidence score (0–100%).
5. **Prioritize**: regions are ranked **HIGH / MEDIUM / LOW** from confidence and size, so
   an operator can triage the most important finds first.
6. **Report**: results are shown as annotated images and a sortable detection table.

## Project structure

```
abysscan_mvp/
├── app.py            # Streamlit UI
├── detection.py       # Core pipeline: preprocessing, Isolation Forest detection,
│                       #   region extraction, priority scoring, synthetic image generator
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
# from inside the abysscan_mvp/ folder
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (default: http://localhost:8501). No dataset or
GPU required.

## Demo instructions

1. On first load, the app already shows a synthetic sonar image with a few embedded
   objects and their detections — nothing to configure.
2. Click **"Use built-in synthetic sonar image"** anytime to generate a fresh one (change
   the seed in the sidebar for a different layout).
3. Turn on **"Show ground-truth markers"** to overlay small cyan crosses where the
   synthetic objects were actually placed, next to the model's own bounding boxes — good
   for showing judges the detector is finding real planted targets, not just drawing
   random boxes.
4. Move the **Sensitivity** slider up to flag more regions (lower detection threshold), or
   down for fewer, higher-confidence-only regions.
5. To demo on your own imagery, upload a `.png`/`.jpg`/`.tif` side-scan sonar image via the
   sidebar uploader — the same pipeline runs unchanged.

## Notes on the detection approach

- This is deliberately **unsupervised**: Isolation Forest is fit fresh on each image's own
  block statistics, so it needs no labeled debris/anomaly dataset to work — useful for a
  hackathon MVP where labeled sonar training data is hard to come by.
- The center **nadir gap** (the dark water-column strip every side-scan sonar image has
  down the middle) is masked out before detection, matching standard side-scan sonar
  processing practice, so it isn't mistaken for a giant anomaly.
- Because it's unsupervised, this stage doesn't *classify* what an object is (drum, net,
  wreck fragment) — it only flags "this doesn't look like normal seabed." Turning this into
  a full classification system is the natural next step (see below).

## Extending this MVP

- **Real datasets**: swap the synthetic generator for images from open side-scan sonar
  datasets (e.g. AI4Shipwrecks, SeabedObjects-KLSG) — just feed them through the uploader,
  no pipeline changes needed.
- **Supervised classification**: once you have labeled data, add a YOLOv8 (or similar)
  detector fine-tuned on debris/wreck classes, and use its output alongside or instead of
  the Isolation Forest stage for a labeled category (not just an anomaly flag).
- **Batch mode**: loop the pipeline over a folder of survey tiles and export one combined
  detection table/CSV per transect.
- **Geotagging**: if your sonar tiles carry navigation metadata (lat/lon per ping), attach
  it to each detected region for direct mapping.
