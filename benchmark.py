import torch
import time
from thop import profile
from model import HybridSegmentationNet

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # New-Netのインスタンス化
    model = HybridSegmentationNet().to(device)
    model.eval()

    # 入力画像サイズ（バッチサイズ1, 3チャンネル, 256x256）
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 1. パラメータ数と計算量（FLOPs）の計測
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    flops = macs * 2  # MACsをFLOPsに近似換算

    print("=== New-Net (Hybrid Segmentation) Benchmark ===")
    print(f"Parameters : {params / 1e6:.2f} M ")
    print(f"Compute    : {flops / 1e9:.2f} GFLOPs")

    # 2. 推論速度（Inference Time / FPS）の計測
    # GPUのウォームアップ（初期化のオーバーヘッドを排除）
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    num_runs = 100
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000
    fps = 1000 / avg_time_ms
    
    print(f"Speed      : {avg_time_ms:.2f} ms / image")
    print(f"FPS        : {fps:.2f} frames per second\n")

if __name__ == "__main__":
    run_benchmark()