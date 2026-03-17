import os
import csv
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import HybridSegmentationNet, HybridLoss

# ==========================================
# 評価指標の計算関数
# ==========================================
def calc_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection / (union + 1e-5)).item()

# ==========================================
# メイン学習ループ
# ==========================================
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 保存先ディレクトリの作成 (weights/ がない場合は作成)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 1. データセットとデータローダーの準備
    train_img_dir = os.path.join(args.data_root, "train/images")
    train_mask_dir = os.path.join(args.data_root, "train/masks")
    val_img_dir = os.path.join(args.data_root, "val/images")
    val_mask_dir = os.path.join(args.data_root, "val/masks")

    train_ds = SegmentationDataset(train_img_dir, train_mask_dir, is_train=True)
    val_ds = SegmentationDataset(val_img_dir, val_mask_dir, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. モデル、損失関数、オプティマイザの初期化
    # 【追加】引数からフラグを受け取り、モデルに渡す
    use_coord_attn = not args.disable_coord_attn
    use_gate = not args.disable_gate
    model = HybridSegmentationNet(use_coord_attn=use_coord_attn, use_gate=use_gate).to(device)
    
    criterion = HybridLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = 0.0

    # CSVログファイルの初期化（ヘッダーの書き込み）
    with open(args.log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_IoU', 'Val_Loss', 'Val_IoU'])

    # 3. エポックループ
    for epoch in range(args.epochs):
        
        # --- Train Phase ---
        model.train()
        train_loss, train_iou = 0.0, 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            preds, gate, deep_sup_masks = model(images)
            loss, l_seg, l_sparse = criterion(preds, masks, gate, deep_sup_masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calc_iou(preds, masks) 

        # --- Validation Phase ---
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                
                preds, gate, deep_sup_masks = model(images)
                loss, l_seg, l_sparse = criterion(preds, masks, gate, deep_sup_masks)
                
                val_loss += loss.item()
                val_iou += calc_iou(preds, masks)

        # --- エポック結果の計算と出力 ---
        t_loss = train_loss / len(train_loader)
        t_iou = train_iou / len(train_loader)
        v_loss = val_loss / len(val_loader)
        v_iou = val_iou / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f}, IoU: {t_iou:.4f} | Val Loss: {v_loss:.4f}, IoU: {v_iou:.4f}")

        # CSVログファイルへの追記
        with open(args.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, t_loss, t_iou, v_loss, v_iou])

        # --- ベストモデルの保存 ---
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./dataset") 
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    # デフォルトの保存先を weights/ 以下に変更
    parser.add_argument("--save_path", type=str, default="weights/hybrid_net_v3_best.pth")
    parser.add_argument("--log_path", type=str, default="weights/log.csv") 
    
    # 【追加】アブレーションスタディ用のフラグ
    parser.add_argument("--disable_coord_attn", action="store_true", help="Coordinate Attentionを無効化")
    parser.add_argument("--disable_gate", action="store_true", help="Attention GateとSoft Gateを無効化")
    
    args = parser.parse_args()
    main(args)