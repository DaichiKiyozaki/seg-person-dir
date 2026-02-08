# seg-person-dir - 同方向/逆方向 歩行者セグメンテーション

## 背景

- 歩行者のセグメンテーション+向き推定の画像認識が必要だった
- 初期の想定構成は以下
    - yolo-seg
        - 人の検出
    - [MEBOW](https://github.com/ChenyanWu/MEBOW)
        - 向きの推定
        - （このモデルは人の角度を出力するが、今回の要件では同方向かそうでないかの判定だけで良い）
- 上記の構成は2モデル構成で、MEBOWは歩行者一人一人に対して推論するため重い
- → YOLOをファインチューニングして「人検出＋向き推定」を1モデルで実現したい

## ディレクトリ構成

```
seg-person-dir/
├── README.md               # このファイル
├── scripts/
│   ├── predict.py          # 推論スクリプト
│   ├── visualize_labels.py # ラベル可視化スクリプト
│   └── 01_make_dataset_frontback_yoloseg.ipynb  # データセット生成ノートブック
├── data/
│   ├── raw/               # COCO抽出画像（ゲート通過のみ）
│   └── dataset_frontback_yoloseg/
│       ├── images/        # 学習用画像
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/        # YOLO-Segラベル
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── reports/        # CSVレポート
│       └── data.yaml       # YOLO設定ファイル（ノートブックで生成）
├── runs/                  # Ultralytics学習出力
```

## クラス定義

| Class ID | Name | 説明 |
|----------|------|------|
| 0 | same-dir-person | 同方向歩行者（カメラと同じ方向を向いている、角度 ≤45° または ≥315°）|
| 1 | ops-dir-person | 逆方向歩行者（カメラと逆方向を向いている）|

## セットアップ

### 1. 依存関係のインストール

```bash
python -m pip install -r requirement.txt
```

### 2. データセット準備

1. COCOアノテーションをダウンロード
2. `scripts/01_make_dataset_frontback_yoloseg.ipynb` を実行してデータセットを生成

## 使用方法

### 学習

**Ultralytics CLI:**
```bash
# 事前に scripts/01_make_dataset_frontback_yoloseg.ipynb を最後まで実行して
# data/dataset_frontback_yoloseg/data.yaml を生成しておく
cd seg-person-dir

yolo segment train data=data/dataset_frontback_yoloseg/data.yaml model=yolo26s-seg.pt imgsz=512 epochs=50 batch=12 project=./runs/segment name=frontback
```

### 推論

**画像/ディレクトリに対して:**
```bash
python scripts/predict.py `
    --weights runs/segment/train_local/weights/best.pt `
    --source data/dataset_frontback_yoloseg/images/val `
    --save --out pred_out
```

**Webカメラでリアルタイム:**
```bash
python scripts/predict.py --weights best.pt --source 0 --show
```

**動画ファイル:**
```bash
python scripts/predict.py --weights best.pt --source video.mp4 --save --show
```

### ラベル可視化（デバッグ用）

```bash
python scripts/visualize_labels.py `
    --images data/dataset_frontback_yoloseg/images/val `
    --labels data/dataset_frontback_yoloseg/labels/val `
    --out label_vis_out --max 100
```

## 参考

- https://dev.classmethod.jp/articles/yolov8-instance-segmentation/
- https://docs.ultralytics.com/ja/models/yolo26/

## 処理フロー概要

1. COCOアノテーションを読み込み
2. **キーポイントゲートで画像IDをフィルタ**（画像はまだダウンロードしない）
3. フィルタ通過IDのみダウンロード
4. 歩行者seg（マスク）はCOCOのGTを使う
5. 歩行者向き判別（MEBOW）で same/ops を付与
6. YOLO-seg形式ラベル（ポリゴン）へ変換
7. 学習（YOLO-Seg）
