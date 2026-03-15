import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

# ==========================================
# Deformable Attention モジュール
# ==========================================
class SimpleDeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_points=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # オフセット予測: 各クエリに対して、num_heads * num_points 個の2次元オフセット(Δx, Δy)を予測
        self.offset_proj = nn.Linear(embed_dim, num_heads * num_points * 2)
        
        # アテンション重み予測: サンプリングポイントに対する重要度を予測
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_points)
        
        # Valueの線形変換と最終出力の線形変換
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, h, w):
        B, N, C = x.shape

        # 1. 基準点 (Reference Points) の生成: [0, 1] の範囲で正規化
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=x.device),
            torch.linspace(0, 1, w, device=x.device),
            indexing='ij'
        )
        ref_points = torch.stack([grid_x, grid_y], dim=-1).reshape(N, 2)
        ref_points = ref_points.unsqueeze(0).unsqueeze(2).unsqueeze(3) # (1, N, 1, 1, 2)

        # 2. オフセットとアテンション重みの計算
        # view の代わりに reshape を使用してメモリ配置エラーを回避
        offsets = self.offset_proj(x).reshape(B, N, self.num_heads, self.num_points, 2)
        weights = self.weight_proj(x).reshape(B, N, self.num_heads, self.num_points)
        weights = F.softmax(weights, dim=-1) # サンプリングポイント間で正規化

        # サンプリング座標 = 基準点 + オフセット (grid_sampleの仕様に合わせて[-1, 1]にスケール変換)
        sample_coords = ref_points + offsets
        sample_coords = sample_coords * 2.0 - 1.0 

        # 3. Valueの空間サンプリング
        v = self.value_proj(x)
        v_spatial = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        
        # reshape を使用して安全に次元変換 (B * num_heads, head_dim, h, w)
        v_spatial = v_spatial.reshape(B * self.num_heads, self.head_dim, h, w)
        
        # 座標テンソルもヘッドごとに分割
        sample_coords_grouped = sample_coords.permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N, self.num_points, 2)

        # F.grid_sampleによる微分可能なバイリニア補間サンプリング
        sampled_v = F.grid_sample(v_spatial, sample_coords_grouped, mode='bilinear', align_corners=False)
        
        # 元の次元構造に復元
        sampled_v = sampled_v.reshape(B, self.num_heads, self.head_dim, N, self.num_points)

        # 4. アテンション重みによる集約
        weights = weights.permute(0, 2, 1, 3).unsqueeze(2)
        weighted_v = torch.sum(sampled_v * weights, dim=-1)

        # 元の系列テンソルに復元
        out = rearrange(weighted_v, 'b head d n -> b n (head d)')
        out = self.output_proj(out)

        return out

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
        self.attn = SimpleDeformableAttention(embed_dim, num_heads=num_heads, num_points=8)
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
        h = w = x_stem.shape[2]
        
        x_seq = rearrange(x_stem, 'b c h w -> b (h w) c')
        x_norm = self.norm1(x_seq)
        
        attn_out = self.attn(x_norm, h, w)
        z_attn = x_seq + attn_out
        
        raw_score = self.score_proj(z_attn)
        gate = self.alpha + (1.0 - self.alpha) * torch.sigmoid(raw_score)
        x_masked = z_attn * gate
        
        x_restored = rearrange(x_masked, 'b (h w) c -> b c h w', h=h, w=w)
        
        encoder_features = self.encoder(x_restored)
        out_mask = self.decoder(encoder_features)
        
        return out_mask, gate

# ==========================================
# 提案手法専用のカスタム損失関数
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, lambda_sparse=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_sparse = lambda_sparse

    def forward(self, pred_mask, true_mask, gate_values):
        l_seg = self.bce(pred_mask, true_mask)
        l_sparse = torch.mean(gate_values)
        total_loss = l_seg + (self.lambda_sparse * l_sparse)
        return total_loss, l_seg, l_sparse