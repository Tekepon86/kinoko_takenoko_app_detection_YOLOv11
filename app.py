import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from pathlib import Path

# Hugging Face Hub から best.pt をダウンロード（キャッシュされる）
weights_path = hf_hub_download(
    repo_id="Tetsushi86/kinoko-takenoko-v11",  # ここを自分のHFリポ名に
    filename="kinoko-takenoko-v11.pt"
)

# 念のためサイズチェック（破損防止）
assert Path(weights_path).exists() and Path(weights_path).stat().st_size > 1_000_000

# YOLO モデル読み込み
model = YOLO(weights_path)

# 🔷 Webカメラ起動
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔷 YOLO11推論
    results = model(frame, coef=0.4)

    # 🔷 結果画像を取得（YOLOv8/11ではplot()を使う）
    annotated_frame = results[0].plot()

    # 🔷 ウィンドウに表示
    cv2.imshow('YOLO11 Realtime Detection', annotated_frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔷 終了処理
cap.release()
cv2.destroyAllWindows()






