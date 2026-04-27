import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# 自作モジュールのインポート
from model import HybridSegmentationNet, HybridLoss
from dataset import SegmentationDataset

def calculate_iou(pred, target):
    """IoU (Intersection over Union) の計算"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_iou = 0, 0
    for images, masks in tqdm(dataloader, desc="  Train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        pred_mask, gate_values = model(images)
        loss, l_bce, l_dice, l_sparse = criterion(pred_mask, masks, gate_values)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += calculate_iou(pred_mask, masks)
    return total_loss / len(dataloader), total_iou / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="  Val  ", leave=False):
            images, masks = images.to(device), masks.to(device)
            pred_mask, gate_values = model(images)
            loss, _, _ = criterion(pred_mask, masks, gate_values)
            total_loss += loss.item()
            total_iou += calculate_iou(pred_mask, masks)
    return total_loss / len(dataloader), total_iou / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = SegmentationDataset(os.path.join(args.data_root, "train/images"), os.path.join(args.data_root, "train/masks"))
    val_ds = SegmentationDataset(os.path.join(args.data_root, "val/images"), os.path.join(args.data_root, "val/masks"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HybridSegmentationNet(alpha=args.alpha).to(device)
    criterion = HybridLoss(lambda_sparse=args.lambda_sparse)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # ログファイルの準備
    log_path = os.path.join(args.save_dir, "learning_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'val_loss', 'val_iou'])

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        t_loss, t_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_iou = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f}, IoU: {t_iou:.4f} | Val Loss: {v_loss:.4f}, IoU: {v_iou:.4f}")

        # ログの書き込み
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, t_loss, t_iou, v_loss, v_iou])

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--save_dir", type=str, default="./weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda_sparse", type=float, default=0.1)
    main(parser.parse_args())