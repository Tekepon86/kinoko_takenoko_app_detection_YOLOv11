import os, io
os.environ["YOLO_CONFIG_DIR"] = "/tmp"  # Ultralyticsの設定警告を抑制（Cloud向け）

import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageOps
from collections import Counter

st.title("きのこ/たけのこ 分類アプリ（YOLOv11・静止画/Cloud対応）")
st.caption("Weights from Hugging Face Hub, PILで描画（OpenCV不要）")

@st.cache_resource
def load_model():
    # Hugging Face Hub から best.pt を取得（キャッシュされる）
    weights_path = hf_hub_download(
        repo_id="Tetsushi86/kinoko-takenoko-v11",  # 例: "taro/yolov11-kinoko-takenoko"
        filename="kinoko-takenoko-v11.pt",
    )
    return YOLO(weights_path)

def prepare_image(img: Image.Image, max_w=1280):
    # スマホEXIFの向き補正＋軽量化のため縮小
    img = ImageOps.exif_transpose(img)
    if img.width > max_w:
        h = int(img.height * max_w / img.width)
        img = img.resize((max_w, h))
    return img

def infer_and_draw(model: YOLO, img: Image.Image, conf=0.4):
    res = model.predict(img, conf=conf, verbose=False)[0]
    names = model.names
    labels = [names[int(b.cls)] for b in res.boxes]

    # PILで矩形とラベルを描画（OpenCV依存なし）
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls = names[int(b.cls)]
        cf  = float(b.conf)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), f"{cls} {cf:.2f}", fill=(0, 255, 0))
    return labels, out

model = load_model()

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = prepare_image(image)
    st.image(image, caption="入力画像", use_container_width=True)

    with st.spinner("検出中..."):
        labels, vis = infer_and_draw(model, image, conf=0.4)

    if not labels:
        st.warning("認識できませんでした。")
    else:
        st.subheader("検出された物体:")
        for k, v in Counter(labels).items():
            st.write(f"- {k}: {v}個")
        st.image(vis, caption="検出結果", use_container_width=True)
