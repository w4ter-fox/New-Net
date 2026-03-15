import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# セグメンテーション用カスタムデータセット
# ==========================================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256)):
        """
        image_dir: 入力画像が保存されているディレクトリのパス
        mask_dir: 正解マスク画像が保存されているディレクトリのパス
        image_size: モデルに入力する解像度 (H, W)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        
        # ディレクトリ内のファイル名リストを取得（アルファベット順にソートしてペアを担保）
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
        # 入力画像とマスク画像の数が一致しているかチェック
        assert len(self.images) == len(self.masks), "画像とマスクのファイル数が一致しません。"

        # 入力画像用の変換処理 (リサイズ -> テンソル化 -> 0.0~1.0に正規化)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
        # マスク画像用の変換処理 (リサイズ -> テンソル化)
        # マスクは通常グレースケール(1チャンネル)で読み込む
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # データセットの総数を返す
        return len(self.images)

    def __getitem__(self, idx):
        # 1. パスの取得
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # 2. 画像の読み込み (RGB形式とグレースケール形式)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # 3. 前処理の適用
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        
        # マスクのピクセル値を0.0と1.0のバイナリ(二値)に厳密化
        mask = (mask > 0.5).float()
        
        return image, mask

# ==========================================
# テスト実行ブロック (ダミーデータを用いた動作確認)
# ==========================================
if __name__ == "__main__":
    import numpy as np
    import shutil
    
    # 1. テスト用のダミーディレクトリと画像を生成
    test_img_dir = "./dummy_images"
    test_mask_dir = "./dummy_masks"
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    print("ダミーデータを生成中...")
    for i in range(3): # 3枚のダミー画像を生成
        # ランダムなノイズ画像 (RGB)
        img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(os.path.join(test_img_dir, f"image_{i:03d}.png"))
        
        # ランダムなマスク画像 (グレースケール)
        mask_array = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
        Image.fromarray(mask_array).save(os.path.join(test_mask_dir, f"image_{i:03d}.png"))

    # 2. データセットとデータローダーの初期化
    dataset = SegmentationDataset(image_dir=test_img_dir, mask_dir=test_mask_dir, image_size=(256, 256))
    
    # バッチサイズ2でデータローダーを作成
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"データセットの総画像数: {len(dataset)}")
    
    # 3. バッチの取り出しとShapeの確認
    for batch_idx, (images, masks) in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Images Shape (B, C, H, W): {images.shape}")
        print(f"Masks Shape (B, C, H, W):  {masks.shape}")
        print(f"Images Data Type: {images.dtype}, Min: {images.min():.2f}, Max: {images.max():.2f}")
        print(f"Masks Data Type:  {masks.dtype}, Min: {masks.min():.2f}, Max: {masks.max():.2f}")
        break # 1バッチ目のみ確認して終了
        
    # 4. 後片付け (ダミーディレクトリの削除)
    shutil.rmtree(test_img_dir)
    shutil.rmtree(test_mask_dir)
    print("\n--- データローダーのテスト完了 ---")