# seg-person-dir

https://qiita.com/kiyokiyomin/items/0d6c524522faeebb336c


## 背景

- 研究要件として、歩行者のセグメンテーションと、向き推定（同方向/逆方向）が必要
- 当初の構成は以下
    - yolo-seg
        - 人の検出
    - [MEBOW](https://github.com/ChenyanWu/MEBOW)
        - 向きの推定
        - （このモデルは5°単位で人の角度を出力するが、今回の要件では同方向か逆方向かの判定だけで良い）
- しかし、上記の構成は、2モデル構成のうえ、MEBOWは歩行者一人一人に対して推論するため重い
- → YOLOをファインチューニングして「人検出＋向き推定」を1モデルで実現したい

## ディレクトリ構成

```
seg-person-dir/
├── README.md               # このファイル
├── MEBOW/                  # 向き推定用（本リポジトリ直下にcloneする）
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

### 1.5 MEBOW の配置（必須）

データセット生成ノートブックでは、歩行者の向き推定に MEBOW を使用します。
そのため、MEBOW を **このリポジトリ直下**の `MEBOW/` に配置してください。

また、推論に使用する重み `seg-person-dir/MEBOW/models/model_hboe.pth` は MEBOW リポジトリに同梱されていないため、
クローン後に MEBOW の GitHub リポジトリで配布されている **Trained HBOE model** を別途ダウンロードして配置してください。

```bash
cd seg-person-dir
git clone https://github.com/ChenyanWu/MEBOW.git
```

配置先（期待パス）:

- `seg-person-dir/MEBOW/models/model_hboe.pth`

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

## 処理フロー概要

1. COCOアノテーションを読み込み
2. **キーポイントゲートで画像IDをフィルタ**（画像はまだダウンロードしない）
3. フィルタ通過IDのみダウンロード
4. 歩行者seg（マスク）はCOCOのGTを使う
5. 歩行者向き判別（MEBOW）で same/ops を付与
6. YOLO-seg形式ラベル（ポリゴン）へ変換
7. 学習（YOLO-Seg）
