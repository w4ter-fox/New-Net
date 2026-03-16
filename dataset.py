import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train # 学習フェーズかどうかのフラグ
        
        # 隠しファイルを除外してリスト化
        self.images = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    def __len__(self):
        return len(self.images)

    def transform(self, image, mask):
        # 1. リサイズ (常に適用)
        image = TF.resize(image, (256, 256))
        # マスクは境界がぼやけないようにニアレストネイバー法でリサイズ
        mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

        # 2. データ拡張 (学習データにのみランダム適用)
        if self.is_train:
            # ランダム水平反転
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # ランダム垂直反転
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # ランダム回転 (-15度 ~ 15度)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 色調変更 (画像にのみ適用し、マスクには適用しない)
            if random.random() > 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)

        # 3. Tensor化と2値化
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()

        return image, mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 同期されたTransformを適用
        image, mask = self.transform(image, mask)

        return image, mask