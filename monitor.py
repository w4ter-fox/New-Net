import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def live_monitor(log_path="weights/learning_log.csv", interval=10):
    """
    学習ログを監視し、リアルタイムでグラフを更新する
    interval: 更新間隔（秒）
    """
    print(f"Monitoring started. Reading from {log_path}...")
    print("Press Ctrl+C to stop monitoring.")

    plt.ion() # インタラクティブモードをオン（画面を閉じずに更新可能にする）
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    while True:
        if os.path.exists(log_path):
            try:
                # ログファイルの読み込み
                df = pd.read_csv(log_path)
                
                if len(df) > 0:
                    ax1.clear()
                    ax2.clear()

                    # --- Loss プロット (左軸) ---
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss', color='tab:red')
                    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='tab:red', linestyle='--')
                    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='tab:red')
                    ax1.tick_params(axis='y', labelcolor='tab:red')

                    # --- IoU プロット (右軸) ---
                    ax2.set_ylabel('IoU', color='tab:blue')
                    ax2.plot(df['epoch'], df['train_iou'], label='Train IoU', color='tab:blue', linestyle='--')
                    ax2.plot(df['epoch'], df['val_iou'], label='Val IoU', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')

                    plt.title(f'Live Training Progress (Latest Epoch: {df["epoch"].max()})')
                    
                    # 凡例の設定
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.1) # 描画を反映
                    
                    # 途中経過を画像としても保存
                    plt.savefig("learning_curve_live.png")
            
            except Exception as e:
                # 学習スクリプトが書き込み中のときに読み込むとエラーが出ることがあるためパス
                pass
        else:
            print(f"Waiting for log file: {log_path}...")

        time.sleep(interval)

if __name__ == "__main__":
    live_monitor()