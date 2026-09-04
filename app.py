import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detection import (
    generate_synthetic_sonar_image,
    preprocess,
    detect_anomalies,
    extract_regions,
    assign_priority,
    draw_regions,
    regions_to_dataframe,
)

st.set_page_config(page_title="ABYSSCAN — Sonar Anomaly Detection", layout="wide")

st.title("ABYSSCAN")
st.caption(
    "Sonar image → preprocessing → unsupervised AI anomaly detection → "
    "highlighted regions → confidence → priority"
)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("1. Input")
    uploaded = st.file_uploader(
        "Upload a side-scan sonar image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    )
    use_synthetic = st.button("Use built-in synthetic sonar image", use_container_width=True)
    seed = st.number_input("Synthetic seed", min_value=0, max_value=9999, value=42, step=1)

    st.header("2. Detection settings")
    block_size = st.select_slider("Analysis block size (px)", options=[8, 16, 24, 32], value=16)
    sensitivity = st.slider("Sensitivity (expected anomaly %)", 2, 25, 8)
    min_region_blocks = st.slider("Min region size (blocks)", 1, 6, 2)
    mask_nadir = st.checkbox("Mask nadir / center water-column gap", value=True)

    st.header("3. Display")
    show_ground_truth = st.checkbox("Show ground-truth markers (synthetic image only)", value=False)

# ---------------------------------------------------------- image loading --
if "gray_img" not in st.session_state:
    st.session_state.gray_img = None
    st.session_state.gt_objects = None

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("L")
    st.session_state.gray_img = np.array(pil_img)
    st.session_state.gt_objects = None
elif use_synthetic:
    img, objs = generate_synthetic_sonar_image(seed=int(seed))
    st.session_state.gray_img = img
    st.session_state.gt_objects = objs
elif st.session_state.gray_img is None:
    # first load: default to the synthetic image so the app is demo-able
    # with zero setup, even with no real dataset on hand
    img, objs = generate_synthetic_sonar_image(seed=int(seed))
    st.session_state.gray_img = img
    st.session_state.gt_objects = objs

gray = st.session_state.gray_img

# --------------------------------------------------------------- pipeline --
denoised, enhanced = preprocess(gray)

anomaly_grid, conf_grid = detect_anomalies(
    enhanced,
    block_size=block_size,
    contamination=sensitivity / 100,
    mask_nadir=mask_nadir,
)
regions = extract_regions(anomaly_grid, conf_grid, block_size, min_blocks=min_region_blocks)
regions = assign_priority(regions)
result_img = draw_regions(gray, regions)

if show_ground_truth and st.session_state.gt_objects:
    for obj in st.session_state.gt_objects:
        cv2.drawMarker(
            result_img, (obj["x"], obj["y"]), (255, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1,
        )

# ------------------------------------------------------------------ views --
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Original")
    st.image(gray, use_container_width=True, clamp=True)
with col2:
    st.subheader("2. Preprocessed (denoise + CLAHE)")
    st.image(enhanced, use_container_width=True, clamp=True)
with col3:
    st.subheader("3. Detected regions")
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Regions flagged", len(regions))
m2.metric("HIGH priority", sum(1 for r in regions if r["priority"] == "HIGH"))
m3.metric("MEDIUM priority", sum(1 for r in regions if r["priority"] == "MEDIUM"))
m4.metric("LOW priority", sum(1 for r in regions if r["priority"] == "LOW"))

st.subheader("Detection table")
if regions:
    st.dataframe(regions_to_dataframe(regions), use_container_width=True, hide_index=True)
else:
    st.info("No anomalies flagged at the current sensitivity — try raising the sensitivity slider.")

with st.expander("How this pipeline works"):
    st.markdown(
        """
1. **Preprocessing** — Non-local means denoising removes sonar speckle noise, then CLAHE
   (Contrast Limited Adaptive Histogram Equalization) boosts local contrast so faint
   returns become visible without blowing out bright ones.
2. **Feature extraction** — the image is split into fixed-size blocks; each block gets a
   feature vector (mean intensity, intensity std-dev, max intensity, edge energy from a
   Laplacian filter).
3. **Unsupervised anomaly detection** — an Isolation Forest is fit *on this image's own
   blocks*, with no labeled training data. Blocks whose features are unusually easy to
   isolate relative to the rest of the seabed get flagged as anomalous.
4. **Region extraction** — flagged blocks are merged into connected regions (via
   connected-component labeling), each scored by its average anomaly confidence and
   sized by how many blocks it spans.
5. **Priority** — regions are ranked HIGH / MEDIUM / LOW from confidence and size, so an
   operator can triage the highest-risk finds first.

The nadir/water-column gap down the center of a side-scan swath is masked out before
detection, since it's a known imaging artifact rather than a target.
        """
    )

st.caption(
    "Built-in synthetic sonar image lets you demo this with zero dataset setup. "
    "Swap in real side-scan sonar imagery (e.g. AI4Shipwrecks, SeabedObjects-KLSG) "
    "via the uploader above — the pipeline runs unchanged."
)
