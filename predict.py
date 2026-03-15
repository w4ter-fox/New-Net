import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import HybridSegmentationNet

def predict_batch(model_path="weights/best_model.pth", val_dir="dataset/val/images", output_dir="results", num_images=10):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridSegmentationNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    val_files = sorted(os.listdir(val_dir))[:num_images]

    for i, file_name in enumerate(val_files):
        img_path = os.path.join(val_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_logits, gate_values = model(input_tensor)
            mask = (torch.sigmoid(pred_logits) > 0.5).float().squeeze().cpu().numpy()
            gate = gate_values.squeeze().cpu().reshape(64, 64).numpy()

        # 描画
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(np.array(img.resize((256, 256))))
        axes[0].set_title("Original")
        axes[1].imshow(gate, cmap='hot')
        axes[1].set_title("Attention Gate")
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title("Prediction")
        
        for ax in axes: ax.axis("off")
        plt.savefig(os.path.join(output_dir, f"result_{i}.png"))
        plt.close()
        print(f"Saved: result_{i}.png")

if __name__ == "__main__":
    import numpy as np
    predict_batch()