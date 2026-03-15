import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# 自作モジュールのインポート
from model import HybridSegmentationNet

def predict_and_visualize(image_path, model_path, output_path="prediction_result.png", device='cpu'):
    # 1. モデルの初期化と重みのロード
    model = HybridSegmentationNet().to(device)
    if os.path.exists(model_path):
        # 学習済みの重みを読み込む
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"重みをロードしました: {model_path}")
    else:
        print(f"警告: {model_path} が見つかりません。初期重みで推論を実行します。")
    
    # モデルを推論モードに設定 (DropoutやBatchNormの挙動を固定)
    model.eval()

    # 2. 画像の前処理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    original_img = Image.open(image_path).convert("RGB")
    # (C, H, W) -> バッチ次元を追加して (1, C, H, W) に変換
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # 3. 推論の実行
    with torch.no_grad(): # 勾配計算を無効化してメモリ消費を抑える
        # ネットワークは (out_mask, gate) を返す仕様
        pred_logits, _ = model(input_tensor)
        
        # ロジットを確率(0.0~1.0)に変換し、0.5を閾値として二値化
        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = (pred_prob > 0.5).float()

    # 4. 可視化のためのデータ変換 (テンソルからNumPy配列へ)
    # matplotlibで扱うために次元を (H, W, C) に戻す
    img_np = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    mask_np = pred_mask.squeeze().cpu().numpy()

    # 5. 結果の描画と保存
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # オリジナル画像
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # 予測されたマスク
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    # オーバーレイ (画像の上にマスクを半透明で重ねる)
    axes[2].imshow(img_np)
    axes[2].imshow(mask_np, cmap='jet', alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"推論結果の画像を保存しました: {output_path}")
    plt.close()

# ==========================================
# テスト実行ブロック
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_img_path = "./dummy_test_image.png"
    
    # テスト用のダミー画像を生成
    dummy_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(dummy_array).save(test_img_path)
    
    # 推論の実行 (train.pyで保存した重みを指定)
    predict_and_visualize(
        image_path=test_img_path,
        model_path="weights/hybrid_seg_net_latest.pth",
        output_path="prediction_result.png",
        device=device
    )
    
    # ダミー画像の削除
    os.remove(test_img_path)
    print("--- 推論・可視化テスト完了 ---")