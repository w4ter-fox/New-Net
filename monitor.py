import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def live_monitor(log_path="weights/log.csv", interval=5):
    """
    学習ログを監視し、リアルタイムでグラフを更新する
    """
    print(f"Monitoring started. Reading from {log_path}...")
    print("Press Ctrl+C to stop monitoring.")

    # GUIがない環境（Codespaces等）でも動作するように設定
    plt.switch_backend('Agg') 
    
    while True:
        if os.path.exists(log_path):
            try:
                # ログファイルの読み込み
                df = pd.read_csv(log_path)
                
                # train.pyで定義したカラム名（大文字開始）に合わせて取得
                if len(df) > 0:
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    ax2 = ax1.twinx()

                    # --- Loss プロット (左軸) ---
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss', color='tab:red')
                    ax1.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', color='tab:red', linestyle='--')
                    ax1.plot(df['Epoch'], df['Val_Loss'], label='Val Loss', color='tab:red')
                    ax1.tick_params(axis='y', labelcolor='tab:red')

                    # --- IoU プロット (右軸) ---
                    ax2.set_ylabel('IoU', color='tab:blue')
                    ax2.plot(df['Epoch'], df['Train_IoU'], label='Train IoU', color='tab:blue', linestyle='--')
                    ax2.plot(df['Epoch'], df['Val_IoU'], label='Val IoU', color='tab:blue')
                    ax2.tick_params(axis='y', labelcolor='tab:blue')
                    ax2.set_ylim(0, 1) # IoUは0~1の範囲

                    plt.title(f'Live Training Progress (Latest Epoch: {df["Epoch"].max()})')
                    
                    # 凡例の設定
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                    plt.tight_layout()
                    
                    # グラフを画像として保存（Codespacesではこれを開いて確認する）
                    plt.savefig("learning_curve_live.png")
                    plt.close(fig)
                    
                    print(f"Updated plot: Epoch {df['Epoch'].max()}")
            
            except Exception as e:
                # 書き込み中の読み込みエラーを無視
                pass
        else:
            print(f"Waiting for log file: {log_path}...")

        time.sleep(interval)

if __name__ == "__main__":
    live_monitor()