import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import shutil  # フォルダ削除用に追加
from model import HybridSegmentationNet

def predict_batch(
    model_path="weights/hybrid_net_v3_best.pth",
    val_dir="dataset/val/images",
    output_dir="results",
    num_images=10
):
    # --- 【追加】results フォルダをリセット ---
    if os.path.exists(output_dir):
        print(f"Resetting {output_dir} directory...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    # ----------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. モデルの初期化
    model = HybridSegmentationNet().to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 画像リストの取得
    val_files = sorted([f for f in os.listdir(val_dir) if not f.startswith('.')])[:num_images]

    for i, file_name in enumerate(val_files):
        img_path = os.path.join(val_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        
        # 前処理
        input_tensor = TF.to_tensor(TF.resize(img, (256, 256))).unsqueeze(0).to(device)

        with torch.no_grad():
            # v3モデルの出力
            pred_logits, gate_values, _ = model(input_tensor)
            
            mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze().cpu().numpy()
            
            # ゲート値のリシェイプ (64x64)
            gate = gate_values.squeeze().cpu().numpy()
            side_len = int(np.sqrt(gate.size))
            gate = gate.reshape(side_len, side_len)

        # 3. 描画
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(np.array(img.resize((256, 256))))
        axes[0].set_title("Original")
        
        axes[1].imshow(gate, cmap='hot')
        axes[1].set_title("Attention Gate")
        
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Prediction")
        
        for ax in axes:
            ax.axis("off")
        
        # 4. 保存
        save_name = f"result_{file_name}"
        save_path = os.path.join(output_dir, save_name)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    predict_batch()