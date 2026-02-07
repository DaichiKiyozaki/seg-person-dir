# predict.py
# ============================================================================
# YOLO-Seg 推論スクリプト（同方向/逆方向 歩行者セグメンテーション）
# ============================================================================
#
# 概要:
#   - 学習済みYOLO-Segモデルを使って、画像/動画/Webカメラで推論
#   - SAME（同方向）は緑、OPS（逆方向）は青でマスク・バウンディングボックス表示
#
# 使用方法:
#   # 画像に対して推論（保存）
#   python predict.py --weights runs/segment/train_local/weights/best.pt \
#                     --source data/dataset_frontback_yoloseg/images/val \
#                     --save --out pred_out
#
#   # Webカメラでリアルタイム推論
#   python predict.py --weights best.pt --source 0 --show
#
#   # 動画ファイル
#   python predict.py --weights best.pt --source video.mp4 --save --show
# ============================================================================

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ----- クラス定義 -----
# 0: SAME = 同方向歩行者（カメラと同じ方向を向いている）
# 1: OPS  = 逆方向歩行者（カメラと逆方向を向いている = Opposite direction）
CLASS_NAMES = {0: "SAME", 1: "OPS"}


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float = 0.35) -> np.ndarray:
    """
    セグメンテーションマスクを画像に半透明で重ねる
    
    Args:
        img_bgr (np.ndarray): 元画像 (BGR形式, HxWx3)
        mask (np.ndarray): セグメンテーションマスク (HxW, 0/1 or bool)
        color_bgr (tuple): マスクの色 (B, G, R)
        alpha (float): マスクの透明度 (0.0 = 完全透明, 1.0 = 完全不透明)
    
    Returns:
        np.ndarray: マスクを重ねた画像
    """
    # マスクを0/1のバイナリに変換
    m = (mask > 0).astype(np.uint8)
    
    # オーバーレイ用の画像をコピー
    overlay = img_bgr.copy()
    
    # マスク領域に色を適用
    overlay[m == 1] = np.array(color_bgr, dtype=np.uint8)
    
    # 元画像とブレンド（alpha: オーバーレイの重み, 1-alpha: 元画像の重み）
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def resize_mask_to_image(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    マスクを画像サイズにリサイズ
    
    Args:
        mask (np.ndarray): 元のマスク
        target_h (int): 目標の高さ
        target_w (int): 目標の幅
    
    Returns:
        np.ndarray: リサイズされたマスク
    """
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return mask


def draw_results(img_bgr: np.ndarray, result, conf_thres: float = 0.25) -> np.ndarray:
    """
    YOLOの推論結果を画像に描画
    
    処理内容:
        1. セグメンテーションマスクの描画（半透明）
        2. バウンディングボックスの描画
        3. クラス名と信頼度のテキスト描画
    
    Args:
        img_bgr (np.ndarray): 入力画像 (BGR)
        result: YOLO推論結果オブジェクト
        conf_thres (float): 表示する最小信頼度閾値
    
    Returns:
        np.ndarray: 描画済み画像
    
    色の対応:
        - SAME (class 0): 青 (255, 0, 0)
        - OPS  (class 1): 赤 (0, 0, 255)
    """
    vis = img_bgr.copy()
    
    # 検出結果がない場合はそのまま返す
    if result.boxes is None or len(result.boxes) == 0:
        return vis

    boxes = result.boxes
    masks = result.masks

    # NumPy配列に変換
    cls_ids = boxes.cls.cpu().numpy().astype(int)  # クラスID
    confs = boxes.conf.cpu().numpy()               # 信頼度
    
    # マスクデータの取得（存在する場合）
    mask_data = None
    if masks is not None and masks.data is not None:
        mask_data = masks.data.cpu().numpy()

    h, w = vis.shape[:2]  # 画像の高さ・幅

    # ----- 各検出に対して描画 -----
    for i, (cid, conf) in enumerate(zip(cls_ids, confs)):
        # 信頼度が閾値未満はスキップ
        if conf < conf_thres:
            continue
        
        # クラス名と色の決定
        name = CLASS_NAMES.get(cid, str(cid))
        # SAME: 青, OPS: 赤
        color = (255, 0, 0) if cid == 0 else (0, 0, 255)

        # ----- バウンディングボックス座標の取得 -----
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
        # 画像範囲内にクリップ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # ----- セグメンテーションマスクの描画 -----
        if mask_data is not None and i < mask_data.shape[0]:
            mask = mask_data[i]
            # マスクサイズが画像サイズと異なる場合はリサイズ
            mask = resize_mask_to_image(mask, h, w)
            vis = overlay_mask(vis, mask, color_bgr=color, alpha=0.35)

        # ----- バウンディングボックスの描画 -----
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # ----- ラベルテキストの描画 -----
        # 背景が見やすいようにボックスの上に表示
        label_text = f"{name} {conf:.2f}"
        cv2.putText(
            vis, 
            label_text, 
            (x1, max(0, y1 - 7)),  # テキスト位置（ボックスの上）
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7,                    # フォントサイズ
            color, 
            2,                      # 線の太さ
            cv2.LINE_AA             # アンチエイリアス
        )
    
    return vis


def iter_images(path: Path):
    """
    指定パスから画像ファイルをイテレート
    
    Args:
        path (Path): ファイルまたはディレクトリのパス
    
    Yields:
        Path: 画像ファイルのパス
    
    対応拡張子: .jpg, .jpeg, .png, .bmp, .webp
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    if path.is_dir():
        # ディレクトリの場合、再帰的に画像を検索
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in exts:
                yield p
    else:
        # 単一ファイルの場合
        yield path


def process_webcam(model: YOLO, args):
    """
    Webカメラからのリアルタイム推論処理
    
    Args:
        model: YOLOモデル
        args: コマンドライン引数
    """
    cam_id = int(args.source)
    
    # Try DirectShow backend first (better for external cameras on Windows)
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"カメラ {cam_id} を開けませんでした。デバイスマネージャーで確認してください。")

    writer = None
    
    # 保存が有効な場合、VideoWriterを初期化
    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        # FPSが取得できない場合はデフォルト値
        if fps <= 0 or np.isnan(fps):
            fps = 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / "webcam_pred.mp4"), fourcc, fps, (w, h))

    print("Webカメラ推論開始... (ESCで終了)")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        # YOLO推論実行
        res = model.predict(
            frame, 
            imgsz=args.imgsz, 
            conf=args.conf, 
            device=args.device, 
            verbose=False
        )[0]
        
        # 結果を描画
        vis = draw_results(frame, res, conf_thres=args.conf)

        # 表示
        if args.show:
            cv2.imshow("YOLO-Seg Prediction", vis)
            # ESCキーで終了
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        # 保存
        if writer is not None:
            writer.write(vis)

    # リソース解放
    cap.release()
    if writer is not None:
        writer.release()
        print(f"保存完了: {args.out}/webcam_pred.mp4")
    cv2.destroyAllWindows()


def process_video(model: YOLO, src: Path, args):
    """
    動画ファイルの推論処理
    
    Args:
        model: YOLOモデル
        src: 動画ファイルパス
        args: コマンドライン引数
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {src}")

    writer = None
    out_dir = Path(args.out)
    
    # 保存が有効な場合、VideoWriterを初期化
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / f"{src.stem}_pred.mp4"), fourcc, fps, (w, h))

    print(f"動画処理中: {src}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        current_frame += 1
        
        # 進捗表示
        if current_frame % 30 == 0:
            print(f"  処理中: {current_frame}/{frame_count} フレーム")
        
        # YOLO推論
        res = model.predict(
            frame, 
            imgsz=args.imgsz, 
            conf=args.conf, 
            device=args.device, 
            verbose=False
        )[0]
        
        vis = draw_results(frame, res, conf_thres=args.conf)

        if args.show:
            cv2.imshow("YOLO-Seg Prediction", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"保存完了: {out_dir / f'{src.stem}_pred.mp4'}")
    cv2.destroyAllWindows()


def process_images(model: YOLO, src: Path, args):
    """
    画像ファイル/ディレクトリの推論処理
    
    Args:
        model: YOLOモデル
        src: 画像ファイルまたはディレクトリのパス
        args: コマンドライン引数
    """
    out_dir = Path(args.out)
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    for p in iter_images(src):
        img = cv2.imread(str(p))
        if img is None:
            print(f"警告: 読み込めませんでした: {p}")
            continue
        
        # YOLO推論
        res = model.predict(
            img, 
            imgsz=args.imgsz, 
            conf=args.conf, 
            device=args.device, 
            verbose=False
        )[0]
        
        vis = draw_results(img, res, conf_thres=args.conf)

        # 保存
        if args.save:
            out_path = out_dir / f"{p.stem}_pred{p.suffix}"
            cv2.imwrite(str(out_path), vis)
            print(f"保存: {out_path}")

        # 表示
        if args.show:
            cv2.imshow("YOLO-Seg Prediction", vis)
            k = cv2.waitKey(0) & 0xFF
            # ESCで終了
            if k == 27:
                break
    
    cv2.destroyAllWindows()


def main():
    """
    メイン処理
    
    入力ソースの種類に応じて適切な処理関数を呼び出す:
        - 数字 → Webカメラ
        - .mp4/.mov/.avi/.mkv → 動画ファイル
        - それ以外 → 画像ファイル/ディレクトリ
    """
    # ----- 引数パース -----
    ap = argparse.ArgumentParser(
        description="YOLO-Seg 推論スクリプト（同方向/逆方向 歩行者）"
    )
    ap.add_argument("--weights", required=True, help="学習済みモデルのパス (best.pt / last.pt)")
    ap.add_argument("--source", required=True, help="入力ソース (画像/ディレクトリ/動画/0=Webカメラ)")
    ap.add_argument("--imgsz", type=int, default=512, help="推論時の画像サイズ")
    ap.add_argument("--conf", type=float, default=0.45, help="信頼度閾値")
    ap.add_argument("--device", default="0", help="デバイス (0, 1, cpu)")
    ap.add_argument("--show", action="store_true", help="結果をウィンドウ表示")
    ap.add_argument("--save", action="store_true", help="結果を保存")
    ap.add_argument("--out", default="pred_out", help="出力ディレクトリ")
    args = ap.parse_args()

    # ----- モデルロード -----
    print(f"モデルをロード中: {args.weights}")
    model = YOLO(args.weights)

    # ----- 入力ソースの判定と処理 -----
    if args.source.isdigit():
        # Webカメラ
        process_webcam(model, args)
    
    else:
        src = Path(args.source)
        video_exts = {".mp4", ".mov", ".avi", ".mkv"}
        
        if src.suffix.lower() in video_exts:
            # 動画ファイル
            process_video(model, src, args)
        else:
            # 画像ファイル/ディレクトリ
            process_images(model, src, args)


if __name__ == "__main__":
    main()
