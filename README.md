# kinoko_takenoko_app_detection_YOLOv11
きのこの山 vs たけのこの里 — 物体検知のデモ（Streamlit / YOLOv11 / 高精度）。重みは自動DL、スマホ対応。

物体検出アプリ（YOLOv5 → YOLOv11 × Streamlit）  
**独自データ収集 → 学習 → Webデモ** まで一貫して構築したプロジェクトです。

---

## 🚀 デモ
- [Streamlit Cloudはこちら](https://kinokotakenokoappdetectionyolov11.streamlit.app/)  
  （スマートフォンからも利用可能）
- [リアルタイム検出（ローカル）デモ](main/demo/kinotake_realtime.gif)
---

## 📊 成果
- YOLOv5でハイパーパラメータ調整・前処理を実施し課題を特定  
- 知見を活かしてYOLOv11で再学習し、**mAP@0.95を達成**  
- StreamlitによるWeb UIを実装（画像アップロード・カメラ入力 → 検出結果描画）  
- GitHub × Streamlit Cloud連携でスマホから利用可能

---

## 🛠️ 使用技術
- Python（PyTorch, YOLOv5, YOLOv11, OpenCV, Streamlit）
- データ前処理・拡張：Augmentation, Gray化, 学習率・画像サイズ調整
- FastAPI（将来拡張予定）

---
## 🔍 学習・改善プロセス
- **YOLOv5での試行錯誤**  
  - Gray化による形状ベースの学習  
  - Augmentation（左右反転、明度・コントラスト調整など）  
  - ハイパパラ調整（学習率・画像サイズ、`--evolve` 機能）  
- **YOLOv11での成果**  
  - 最新モデルで精度大幅向上（mAP@0.95）  
  - 推論速度・汎化性能も改善  

👉 詳細は [スライド資料](./slides/kinoko-takenoko-process.pdf) をご覧ください

---

## 📌 今後の拡張
- FastAPI による推論API化  
- YOLOv8 との比較実験  
- Magic Eraser 機能の追加（背景除去 × 推論）

---

## 📝 作者
- **Your Name**  
- Data Science / Machine Learning Enthusiast  
- [LinkedIn](https://linkedin.com/in/xxxx) ｜ [Twitter](https://twitter.com/xxxx)


