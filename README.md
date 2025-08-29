# kinoko_takenoko_app_detection_YOLOv11
きのこの山 vs たけのこの里 — 物体検知のデモ（Streamlit / YOLOv11 / 高精度）。重みは自動DL、スマホ対応。

物体検出アプリ（YOLOv5 → YOLOv11 × Streamlit）  
**独自データ収集 → 学習 → Webデモ** まで一貫して構築したプロジェクトです。

---

## 🚀 デモ
- [Streamlit Cloudはこちら](https://kinokotakenokoappdetectionyolov11.streamlit.app/)  
  （スマートフォンからも利用可能）
- [リアルタイム検出（ローカル）デモ](demo/kinotake_realtime_2.gif)
---

## 📊 成果
- YOLOv5でグレースケール化や基本的なハイパラ調整を試し、課題を特定  
- 知見を活かしてYOLOv11で再学習し、**mAP0.95=0.946 を達成**  
- StreamlitによるWeb UIを実装（画像アップロード・カメラ入力 → 検出結果描画）  
- GitHub × Streamlit Cloud連携でスマホから利用可能

---

## 🛠️ 使用技術
- Python（PyTorch, YOLOv5, YOLOv11, OpenCV, Streamlit）

---
## 🔍 学習・改善プロセス
- **YOLOv5での試行**  
  - グレースケール化による形状ベースの学習  
  - ハイパーパラメータ調整（学習率・画像サイズ、--evolveによる探索）
- **YOLOv11での成果**  
  - 最新モデルで精度大幅向上（mAP0.95=0.946）  
  - 推論速度・汎化性能も改善

👉 詳細は [スライド資料](docs/app_slide.pdf)

---

## 📌 今後の拡張
- FastAPI による推論API化  
- YOLOv11のハイパパラメータチューニング  
- Magic Eraser 機能の追加（背景除去 × 推論）


