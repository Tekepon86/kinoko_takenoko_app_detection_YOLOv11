import cv2
from ultralytics import YOLO

# 🔷 YOLO11モデル読み込み
model = model = YOLO('yolov5/runs/train/kinoko_takenoko_all_30_0609/weights/best_v11_250707_default.pt')  # 重みファイル名は自身のものに変更

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

