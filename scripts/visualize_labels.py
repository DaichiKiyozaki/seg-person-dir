# visualize_labels.py
# ============================================================================
# YOLO-Segラベル可視化スクリプト
# ============================================================================
#
# 概要:
#   - YOLO形式のセグメンテーションラベル（.txt）を画像に重ねて可視化
#   - ラベルが正しく付与されているかを視覚的に確認するためのツール
#   - SAME（同方向）は緑、OPS（逆方向）は青で表示
#
# 使用方法:
#   python visualize_labels.py \
#       --images data/dataset_frontback_yoloseg/images/val \
#       --labels data/dataset_frontback_yoloseg/labels/val \
#       --out label_vis_out --max 100
#
# YOLO-Seg ラベル形式:
#   class_id x1 y1 x2 y2 x3 y3 ... xn yn
#   - class_id: クラスID (0=SAME, 1=OPS)
#   - x,y: 正規化された座標 (0.0-1.0)
# ============================================================================

import argparse
from pathlib import Path
import cv2
import numpy as np

# ----- クラスカラー定義 (BGR形式) -----
COLORS = {
    0: (0, 255, 0),   # SAME (同方向) = 緑
    1: (255, 0, 0),   # OPS (逆方向) = 青
}

# クラス名
CLASS_NAMES = {0: "SAME", 1: "OPS"}


def load_yolo_seg_label(txt_path: Path) -> list:
    """
    YOLO-Segフォーマットのラベルファイルを読み込む
    
    YOLO-Seg ラベル形式:
        各行: class_id x1 y1 x2 y2 ... xn yn
        - 座標は0-1に正規化された値
        - 最低3点（6座標）が必要
    
    Args:
        txt_path (Path): ラベルファイルのパス
    
    Returns:
        list: [(class_id, polygon), ...] のリスト
              polygon は Nx2 の numpy配列（正規化座標）
    """
    items = []
    
    # ファイルが存在しない場合は空リストを返す
    if not txt_path.exists():
        return items
    
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    
    for line_num, ln in enumerate(lines, 1):
        ln = ln.strip()
        if not ln:
            continue
        
        parts = ln.split()
        
        # 最低限: class_id + 6座標（3点）が必要
        if len(parts) < 7:
            print(f"警告: {txt_path.name}:{line_num} - 座標が不足しています")
            continue
        
        # クラスIDの取得
        cls_id = int(parts[0])
        
        # 座標の取得（float配列に変換）
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
        
        # 座標数が偶数でない場合はスキップ
        if coords.size % 2 != 0:
            print(f"警告: {txt_path.name}:{line_num} - 座標数が不正です")
            continue
        
        # Nx2のポリゴン形式にリシェイプ
        poly = coords.reshape(-1, 2)
        
        items.append((cls_id, poly))
    
    return items


def denorm_poly(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    正規化されたポリゴン座標を画像座標に変換
    
    Args:
        poly_norm (np.ndarray): 正規化座標 (0.0-1.0) の Nx2 配列
        w (int): 画像の幅
        h (int): 画像の高さ
    
    Returns:
        np.ndarray: 画像座標 (int) の Nx2 配列
    """
    poly = poly_norm.copy()
    
    # x座標: 0-1 → 0-(w-1)
    poly[:, 0] = np.clip(poly[:, 0] * w, 0, w - 1)
    # y座標: 0-1 → 0-(h-1)
    poly[:, 1] = np.clip(poly[:, 1] * h, 0, h - 1)
    
    return poly.astype(np.int32)


def visualize_single_image(img_path: Path, lbl_path: Path, alpha: float = 0.35) -> np.ndarray:
    """
    1枚の画像にラベルを可視化
    
    Args:
        img_path (Path): 画像ファイルのパス
        lbl_path (Path): ラベルファイルのパス
        alpha (float): マスクの透明度
    
    Returns:
        np.ndarray: 可視化された画像 (BGR)
    """
    # 画像の読み込み
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # ラベルの読み込み
    items = load_yolo_seg_label(lbl_path)
    
    # ラベルがない場合は元画像を返す
    if not items:
        return img
    
    # マスク描画用のレイヤー
    vis = img.copy()
    
    # ----- 各オブジェクトのマスクを描画 -----
    for cls_id, poly_norm in items:
        # クラスに対応する色を取得（不明クラスは黄色）
        color = COLORS.get(cls_id, (0, 255, 255))
        
        # 正規化座標を画像座標に変換
        poly = denorm_poly(poly_norm, w, h)
        
        # ポリゴンを塗りつぶし
        cv2.fillPoly(vis, [poly], color)
    
    # ----- 元画像と透過合成 -----
    vis = cv2.addWeighted(vis, alpha, img, 1 - alpha, 0)
    
    # ----- ラベル情報をテキストで表示 -----
    for i, (cls_id, poly_norm) in enumerate(items):
        poly = denorm_poly(poly_norm, w, h)
        color = COLORS.get(cls_id, (0, 255, 255))
        name = CLASS_NAMES.get(cls_id, f"Class{cls_id}")
        
        # ポリゴンの中心点を計算
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        
        # テキスト描画（影付き）
        cv2.putText(vis, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)  # 影
        cv2.putText(vis, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2, cv2.LINE_AA)
        
        # ポリゴンの輪郭を描画
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
    
    return vis


def main():
    """
    メイン処理: ラベル可視化
    
    処理フロー:
        1. 画像ディレクトリから画像を取得
        2. 対応するラベルファイルを読み込み
        3. ラベルを画像に重ねて描画
        4. 出力ディレクトリに保存
    """
    # ----- 引数パース -----
    ap = argparse.ArgumentParser(
        description="YOLO-Segラベル可視化ツール"
    )
    ap.add_argument(
        "--images", required=True, 
        help="画像ディレクトリ (例: .../images/val)"
    )
    ap.add_argument(
        "--labels", required=True, 
        help="ラベルディレクトリ (例: .../labels/val)"
    )
    ap.add_argument(
        "--out", default="label_vis_out", 
        help="出力ディレクトリ"
    )
    ap.add_argument(
        "--max", type=int, default=50, 
        help="可視化する最大画像数"
    )
    ap.add_argument(
        "--alpha", type=float, default=0.35, 
        help="マスクの透明度 (0.0-1.0)"
    )
    args = ap.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_dir = Path(args.out)
    
    # 出力ディレクトリの作成
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 画像ファイルの取得 -----
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    
    # 最大数で制限
    imgs = imgs[:args.max]
    
    if not imgs:
        print(f"エラー: 画像が見つかりません: {img_dir}")
        return

    print(f"可視化開始: {len(imgs)} 枚の画像")
    print(f"  画像: {img_dir}")
    print(f"  ラベル: {lbl_dir}")
    print(f"  出力: {out_dir}")
    print("-" * 50)

    # ----- 各画像を処理 -----
    success_count = 0
    no_label_count = 0
    
    for i, img_path in enumerate(imgs, 1):
        # 対応するラベルファイルのパス（拡張子を.txtに変更）
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        # 可視化
        vis = visualize_single_image(img_path, lbl_path, alpha=args.alpha)
        
        if vis is None:
            print(f"[{i}/{len(imgs)}] スキップ（読み込み失敗）: {img_path.name}")
            continue
        
        # ラベルが存在しない場合のカウント
        if not lbl_path.exists():
            no_label_count += 1
        
        # 保存
        out_path = out_dir / f"{img_path.stem}_labelvis{img_path.suffix}"
        cv2.imwrite(str(out_path), vis)
        success_count += 1
        
        # 進捗表示（10枚ごと）
        if i % 10 == 0 or i == len(imgs):
            print(f"[{i}/{len(imgs)}] 処理完了")

    # ----- 完了メッセージ -----
    print("-" * 50)
    print(f"完了: {success_count} 枚の画像を保存しました")
    if no_label_count > 0:
        print(f"  ※ラベルなし: {no_label_count} 枚")
    print(f"出力先: {out_dir}")


if __name__ == "__main__":
    main()
