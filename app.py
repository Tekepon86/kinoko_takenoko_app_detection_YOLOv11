import cv2
import urllib.request
from ultralytics import YOLO
from pathlib import Path

# 🔷 YOLO11モデル読み込み
weights_path = Path("models/best.pt")
if not weights_path.exists():
    url = "hhttps://huggingface.co/Tetsushi86/kinoko-takenoko-v11/resolve/main/kinoko-takenoko-v11.pt"
    urllib.request.urlretrieve(url, weights_path)
    
model = YOLO(str(weights_path))

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




