import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

# ==========================================
# 1. Deformable Attention モジュール
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
        offsets = self.offset_proj(x).reshape(B, N, self.num_heads, self.num_points, 2)
        weights = self.weight_proj(x).reshape(B, N, self.num_heads, self.num_points)
        weights = F.softmax(weights, dim=-1) # サンプリングポイント間で正規化

        # サンプリング座標 = 基準点 + オフセット (grid_sampleの仕様に合わせて[-1, 1]にスケール変換)
        sample_coords = ref_points + offsets
        sample_coords = sample_coords * 2.0 - 1.0 

        # 3. Valueの空間サンプリング
        v = self.value_proj(x)
        v_spatial = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        
        v_spatial = v_spatial.reshape(B * self.num_heads, self.head_dim, h, w)
        sample_coords_grouped = sample_coords.permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N, self.num_points, 2)

        # F.grid_sampleによる微分可能なバイリニア補間サンプリング
        sampled_v = F.grid_sample(v_spatial, sample_coords_grouped, mode='bilinear', align_corners=False)
        sampled_v = sampled_v.reshape(B, self.num_heads, self.head_dim, N, self.num_points)

        # 4. アテンション重みによる集約
        weights = weights.permute(0, 2, 1, 3).unsqueeze(2)
        weighted_v = torch.sum(sampled_v * weights, dim=-1)

        # 元の系列テンソルに復元
        out = rearrange(weighted_v, 'b head d n -> b n (head d)')
        out = self.output_proj(out)

        return out

# ==========================================
# 2. 軽量デコーダ用モジュール (U-Net Decoder)
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
    def __init__(self, encoder_channels=[16, 24, 40, 112, 960], decoder_channels=[256, 128, 64, 32, 16]):
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
# 3. 統合モデル (New-Net v5.1)
# ==========================================
class HybridSegmentationNet(nn.Module):
    def __init__(self, num_heads=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
        # ImageNetで事前学習済みの標準MobileNetV3を読み込む (pretrained=True)
        self.encoder = timm.create_model('mobilenetv3_large_100', pretrained=True)
        
        # MobileNetV3_largeの blocks[1] の出力チャンネル数は「24」
        self.embed_dim = 24 
        self.norm1 = nn.LayerNorm(self.embed_dim)
        
        # 24チャンネルを受け取るアテンション (24は4の倍数なのでnum_heads=4で割り切れる)
        self.attn = SimpleDeformableAttention(self.embed_dim, num_heads=num_heads, num_points=8)
        self.score_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1)
        )

        # デコーダの初期化 (MobileNetV3の各ブロックの出力次元に厳密に一致させる)
        self.decoder = LightweightUNetDecoder(encoder_channels=[16, 24, 40, 112, 960])

    def forward(self, x):
        # ---------------------------------------------------------
        # 【Stem】: ImageNetの重みで高品質な初期特徴を抽出
        # ---------------------------------------------------------
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        # ※ Colabのtimm仕様に合わせ、ここで self.encoder.act1(x) は呼び出さない
        
        f0 = self.encoder.blocks[0](x)   # (B, 16, H/2, W/2) -> Skip 1
        f1 = self.encoder.blocks[1](f0)  # (B, 24, H/4, W/4) -> Attentionへ
        
        # ---------------------------------------------------------
        # 【Attention】: Deformable Attention + α-スケーリング
        # ---------------------------------------------------------
        h, w = f1.shape[2:]
        f1_seq = rearrange(f1, 'b c h w -> b (h w) c')
        f1_norm = self.norm1(f1_seq)
        
        attn_out = self.attn(f1_norm, h, w)
        z_attn = f1_seq + attn_out
        
        # α-スケーリングを用いた Leaky Gate の計算
        raw_score = self.score_proj(z_attn)
        gate = self.alpha + (1.0 - self.alpha) * torch.sigmoid(raw_score)
        
        f1_masked_seq = z_attn * gate
        f1_masked = rearrange(f1_masked_seq, 'b (h w) c -> b c h w', h=h, w=w) # Skip 2
        
        # ---------------------------------------------------------
        # 【Backbone】: ノイズを遮断した綺麗な特徴を深い層へ
        # ---------------------------------------------------------
        f2 = self.encoder.blocks[2](f1_masked) # (B, 40, H/8, W/8) -> Skip 3
        
        x_f3 = self.encoder.blocks[3](f2)
        f3 = self.encoder.blocks[4](x_f3)      # (B, 112, H/16, W/16) -> Skip 4
        
        x_f4 = self.encoder.blocks[5](f3)      # 出力チャンネル: 160
        f4 = self.encoder.blocks[6](x_f4)      # 出力チャンネル: 960 (B, 960, H/32, W/32) -> Skip 5
        
        # ---------------------------------------------------------
        # 【Decoder】: 空間の復元
        # ---------------------------------------------------------
        out_mask = self.decoder([f0, f1_masked, f2, f3, f4])
        
        return out_mask, gate

# ==========================================
# 4. 提案手法専用 カスタム損失関数
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, lambda_sparse=0.1, lambda_dice=2.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_sparse = lambda_sparse
        self.lambda_dice = lambda_dice

    def forward(self, pred_mask, true_mask, gate_values):
        # BCE Loss
        l_bce = self.bce(pred_mask, true_mask)
        
        # Dice Loss (対象物の形状一致を強烈に促す)
        pred_prob = torch.sigmoid(pred_mask)
        intersection = torch.sum(pred_prob * true_mask)
        union = torch.sum(pred_prob) + torch.sum(true_mask)
        l_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        # Sparse Loss (ゲートを閉じて背景ノイズを削る圧力)
        l_sparse = torch.mean(gate_values)
        
        # 総合 Loss
        total_loss = l_bce + (self.lambda_dice * l_dice) + (self.lambda_sparse * l_sparse)
        
        return total_loss, l_bce, l_dice, l_sparse