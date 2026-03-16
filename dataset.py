import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.is_train = is_train

        # 訓練用（データ拡張あり）
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # 細長い構造の向きやスケール変化に対応
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                # 有機的・不規則な曲がり具合をシミュレート
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                # 画像の正規化 (ImageNet基準)
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        # 検証用（データ拡張なし）
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # 画像の読み込み
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # マスクの読み込み（グレースケール）
        # 値が[0, 1]であることを前提とし、そのままの数値を維持
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # 画像とマスクへ同時に同じ変換を適用
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # ネットワークの出力と損失関数の次元に合わせるため [1, H, W] に変形
        mask = mask.unsqueeze(0)

        return image, mask