import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 作成した自作モジュールをインポート
from model import HybridSegmentationNet, HybridLoss
from dataset import SegmentationDataset

def train_model(image_dir, mask_dir, num_epochs=5, batch_size=4, learning_rate=1e-4, device='cpu'):
    # 1. データセットとデータローダーの準備
    dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, image_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"データセットサイズ: {len(dataset)}枚")
    print(f"使用デバイス: {device}")

    # 2. モデル、損失関数、最適化手法の定義
    model = HybridSegmentationNet().to(device)
    criterion = HybridLoss(lambda_sparse=0.1)
    
    # AdamWオプティマイザ (重み減衰を利用して過学習を防ぐ)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 3. 学習ループ
    for epoch in range(num_epochs):
        model.train() # 学習モードに設定
        epoch_loss = 0.0
        epoch_l_seg = 0.0
        epoch_l_sparse = 0.0
        
        # tqdmでプログレスバーを表示
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, masks in pbar:
                # データをGPU/CPUへ転送
                images = images.to(device)
                masks = masks.to(device)
                
                # 勾配の初期化
                optimizer.zero_grad()
                
                # 順伝播 (Forward)
                pred_mask, gate_values = model(images)
                
                # 損失計算 (Loss)
                loss, l_seg, l_sparse = criterion(pred_mask, masks, gate_values)
                
                # 逆伝播 (Backward)
                loss.backward()
                
                # パラメータ更新 (Optimizer step)
                optimizer.step()
                
                # ログの記録
                epoch_loss += loss.item()
                epoch_l_seg += l_seg.item()
                epoch_l_sparse += l_sparse.item()
                
                # プログレスバーの表示更新
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
        # エポックごとの平均Lossを出力
        avg_loss = epoch_loss / len(dataloader)
        avg_l_seg = epoch_l_seg / len(dataloader)
        avg_l_sparse = epoch_l_sparse / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f} (Seg: {avg_l_seg:.4f}, Sparse: {avg_l_sparse:.4f})")
        
    # 4. 学習済みモデルの保存
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/hybrid_seg_net_latest.pth"
    torch.save(model.state_dict(), save_path)
    print(f"モデルの重みを保存しました: {save_path}")

# ==========================================
# テスト実行ブロック (ダミーデータを用いた学習ループの動作確認)
# ==========================================
if __name__ == "__main__":
    import numpy as np
    import shutil
    from PIL import Image
    
    # GPUが使える場合はGPUを、そうでない場合はCPUを使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ダミーディレクトリと画像の生成
    test_img_dir = "./dummy_train_images"
    test_mask_dir = "./dummy_train_masks"
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    print("学習ループテスト用のダミーデータを生成中...")
    for i in range(8): # 8枚のダミー画像を生成 (Batch=4なら1エポックあたり2イテレーション)
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(os.path.join(test_img_dir, f"image_{i:03d}.png"))
        
        mask_array = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(mask_array).save(os.path.join(test_mask_dir, f"image_{i:03d}.png"))

    # 学習ループの実行 (テストのため2エポックのみ)
    train_model(
        image_dir=test_img_dir, 
        mask_dir=test_mask_dir, 
        num_epochs=2, 
        batch_size=4, 
        learning_rate=1e-4, 
        device=device
    )
    
    # 後片付け
    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_mask_dir)
    print("\n--- 学習パイプラインのテスト完了 ---")