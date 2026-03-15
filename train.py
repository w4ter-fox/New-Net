import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 自作モジュールのインポート
from model import HybridSegmentationNet, HybridLoss
from dataset import SegmentationDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1エポック分の学習を実行"""
    model.train()
    total_loss, total_seg, total_sparse = 0, 0, 0
    
    pbar = tqdm(dataloader, desc="  Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        pred_mask, gate_values = model(images)
        loss, l_seg, l_sparse = criterion(pred_mask, masks, gate_values)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_seg += l_seg.item()
        total_sparse += l_sparse.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader), total_seg / len(dataloader), total_sparse / len(dataloader)

def validate(model, dataloader, criterion, device):
    """検証データでの評価を実行"""
    model.eval()
    total_loss, total_seg, total_sparse = 0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  Val  ", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            pred_mask, gate_values = model(images)
            loss, l_seg, l_sparse = criterion(pred_mask, masks, gate_values)
            
            total_loss += loss.item()
            total_seg += l_seg.item()
            total_sparse += l_sparse.item()
            
    return total_loss / len(dataloader), total_seg / len(dataloader), total_sparse / len(dataloader)

def main(args):
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. データセットとローダー
    train_ds = SegmentationDataset(
        os.path.join(args.data_root, "train/images"),
        os.path.join(args.data_root, "train/masks")
    )
    val_ds = SegmentationDataset(
        os.path.join(args.data_root, "val/images"),
        os.path.join(args.data_root, "val/masks")
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. モデル・損失関数・最適化
    model = HybridSegmentationNet(alpha=args.alpha).to(device)
    criterion = HybridLoss(lambda_sparse=args.lambda_sparse)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Starting Training on {device}...")
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    best_val_loss = float('inf')

    # 3. 学習メインループ
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_seg, train_sparse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_seg, val_sparse = validate(model, val_loader, criterion, device)
        
        print(f"  [Train] Loss: {train_loss:.4f} (Seg: {train_seg:.4f}, Sparse: {train_sparse:.4f})")
        print(f"  [Val]   Loss: {val_loss:.4f} (Seg: {val_seg:.4f}, Sparse: {val_sparse:.4f})")

        # 4. モデルの保存（Best Model更新時）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> Best model saved! (Val Loss: {best_val_loss:.4f})")

    print("\nTraining Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HybridSegNet Training Pipeline")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Dataset root directory")
    parser.add_argument("--save_dir", type=str, default="./weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha value for gate")
    parser.add_argument("--lambda_sparse", type=float, default=0.1, help="Sparsity penalty weight")
    
    args = parser.parse_args()
    main(args)