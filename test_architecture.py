import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

# ==========================================
# デコーダ用モジュール (Lightweight U-Net)
# ==========================================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class LightweightUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32, 16]):
        super().__init__()
        self.b4 = DecoderBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0])
        self.b3 = DecoderBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        self.b2 = DecoderBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])
        self.b1 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])
        
        self.up1 = DecoderBlock(decoder_channels[3], 0, decoder_channels[4])
        self.up2 = DecoderBlock(decoder_channels[4], 0, 16)
        self.up3 = DecoderBlock(16, 0, 16)
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, features):
        f0, f1, f2, f3, f4 = features
        x = self.b4(f4, f3)
        x = self.b3(x, f2)
        x = self.b2(x, f1)
        x = self.b1(x, f0)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        out = self.final_conv(x)
        return out

# ==========================================
# 統合モデル (Proposed Hybrid Segmentation Net)
# ==========================================
class HybridSegmentationNet(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, num_heads=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.score_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        self.encoder = timm.create_model(
            'mobilenetv3_large_100', 
            pretrained=False, 
            in_chans=embed_dim, 
            features_only=True
        )
        
        self.decoder = LightweightUNetDecoder(encoder_channels=[16, 24, 40, 112, 960])

    def forward(self, x):
        x_stem = self.stem(x)
        x_seq = rearrange(x_stem, 'b c h w -> b (h w) c')
        
        x_norm = self.norm1(x_seq)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        z_attn = x_seq + attn_out
        
        raw_score = self.score_proj(z_attn)
        gate = self.alpha + (1.0 - self.alpha) * torch.sigmoid(raw_score)
        x_masked = z_attn * gate
        
        h = w = int(x_masked.shape[1] ** 0.5)
        x_restored = rearrange(x_masked, 'b (h w) c -> b c h w', h=h, w=w)
        
        encoder_features = self.encoder(x_restored)
        out_mask = self.decoder(encoder_features)
        
        # 変更点: 損失関数で計算するために、最終マスクだけでなく内部の gate 値も返す
        return out_mask, gate

# ==========================================
# 提案手法専用のカスタム損失関数
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, lambda_sparse=0.1):
        super().__init__()
        # セグメンテーションの基本損失 (BCE with Logits)
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_sparse = lambda_sparse

    def forward(self, pred_mask, true_mask, gate_values):
        # 1. 主損失 (L_seg)
        l_seg = self.bce(pred_mask, true_mask)
        
        # 2. スパース性ペナルティ (L_sparse)
        # ゲート値の平均を計算。これにより、ネットワークは不要な領域のゲートを閉じるよう促される
        l_sparse = torch.mean(gate_values)
        
        # 総合損失
        total_loss = l_seg + (self.lambda_sparse * l_sparse)
        
        return total_loss, l_seg, l_sparse

# ==========================================
# テスト実行ブロック
# ==========================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 1. ダミーデータの生成 (画像と正解ラベル)
    dummy_image = torch.randn(2, 3, 256, 256)
    dummy_label = torch.empty(2, 1, 256, 256).random_(2) # 0または1のバイナリマスク
    
    # 2. モデルと損失関数の初期化
    model = HybridSegmentationNet()
    criterion = HybridLoss(lambda_sparse=0.1)
    
    print("--- 順伝播 (Forward) ---")
    # モデル出力 (出力マスクと、内部のゲート値の両方を受け取る)
    pred_mask, gate_values = model(dummy_image)
    print(f"Predicted Mask Shape: {pred_mask.shape}")
    print(f"Gate Values Shape: {gate_values.shape}")
    
    print("\n--- 損失計算 (Loss Calculation) ---")
    # 損失の計算
    total_loss, l_seg, l_sparse = criterion(pred_mask, dummy_label, gate_values)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f" ├─ Segmentation Loss: {l_seg.item():.4f}")
    print(f" └─ Sparsity Penalty:  {l_sparse.item():.4f}")
    
    print("\n--- 逆伝播 (Backward) ---")
    # 勾配の計算 (エラーが出なければ微分可能なネットワークとして成立している証明)
    total_loss.backward()
    print("Backward pass completed successfully.")
    
    print("\n--- End-to-End テスト完了 ---")