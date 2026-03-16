import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

# ==========================================
# 1. 提案手法専用のカスタム損失関数（Deep Supervision対応版）
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, lambda_sparse=0.1, lambda_dice=1.0, lambda_deep_sup=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_sparse = lambda_sparse
        self.lambda_dice = lambda_dice
        self.lambda_deep_sup = lambda_deep_sup # Deep Supervisionの重み

    def forward(self, pred_mask, true_mask, gate_values, deep_sup_masks=None):
        # 1. 最終出力に対するセグメンテーション損失（BCE + Dice）
        l_seg_final = self.calc_seg_loss(pred_mask, true_mask)
        
        # 2. Deep Supervision：中間出力に対するセグメンテーション損失
        l_seg_deep = 0.0
        if deep_sup_masks is not None:
            for mask in deep_sup_masks:
                # 正解マスクを中間出力のサイズにリサイズ
                target_rescaled = F.interpolate(true_mask, size=mask.shape[2:], mode='nearest')
                l_seg_deep += self.calc_seg_loss(mask, target_rescaled)
            l_seg_deep /= len(deep_sup_masks) # 平均をとる

        # セグメンテーション全体の損失（最終出力 + 中間出力）
        l_seg = l_seg_final + (self.lambda_deep_sup * l_seg_deep)
        
        # 3. スパース性ペナルティ
        l_sparse = torch.mean(gate_values)
        
        # 総合損失
        total_loss = l_seg + (self.lambda_sparse * l_sparse)
        
        return total_loss, l_seg, l_sparse

    def calc_seg_loss(self, pred, target):
        # BCE Loss
        l_bce = self.bce(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2. * intersection + 1e-5) / (union + 1e-5)
        l_dice = 1.0 - dice_score.mean()
        
        return l_bce + self.lambda_dice * l_dice

# ==========================================
# 2. Deformable Attention モジュール
# ==========================================
class SimpleDeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_points=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        self.offset_proj = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.weight_proj = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, h, w):
        B, N, C = x.shape

        # 参照ポイントの生成
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=x.device),
            torch.linspace(0, 1, w, device=x.device),
            indexing='ij'
        )
        ref_points = torch.stack([grid_x, grid_y], dim=-1).reshape(N, 2)
        ref_points = ref_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # オフセットと重みの予測
        offsets = self.offset_proj(x).reshape(B, N, self.num_heads, self.num_points, 2)
        weights = self.weight_proj(x).reshape(B, N, self.num_heads, self.num_points)
        weights = F.softmax(weights, dim=-1)

        # サンプリング座標の計算
        sample_coords = ref_points + offsets
        sample_coords = sample_coords * 2.0 - 1.0 # normalize to [-1, 1] for grid_sample

        # Valueのプロジェクトと変形
        v = self.value_proj(x)
        v_spatial = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
        v_spatial = v_spatial.reshape(B * self.num_heads, self.head_dim, h, w)
        
        # サンプリング座標の変形
        sample_coords_grouped = sample_coords.permute(0, 2, 1, 3, 4).reshape(B * self.num_heads, N, self.num_points, 2)

        # 特徴量のサンプリング（Deformable Sampling）
        sampled_v = F.grid_sample(v_spatial, sample_coords_grouped, mode='bilinear', align_corners=False)
        sampled_v = sampled_v.reshape(B, self.num_heads, self.head_dim, N, self.num_points)

        # 重み付き和の計算
        weights = weights.permute(0, 2, 1, 3).unsqueeze(2)
        weighted_v = torch.sum(sampled_v * weights, dim=-1)

        # 出力の変換
        out = rearrange(weighted_v, 'b head d n -> b n (head d)')
        out = self.output_proj(out)

        return out

# ==========================================
# 3. デコーダ用モジュール (Attention Gated U-Net)
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

# 修正済み：解像度不一致を解消したAttention Gate
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # g1の解像度をx1の解像度に合わせる
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.attention = attention
        
        if attention:
            self.ag = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        
        self.conv = DepthwiseSeparableConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # 1. アップサンプリング
        x_up = self.up(x)
        
        # 2. スキップ接続の結合
        if skip is not None:
            if x_up.shape[2:] != skip.shape[2:]:
                x_up = F.interpolate(x_up, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Attention Gateでフィルタリング
            if self.attention:
                skip = self.ag(g=x, x=skip) # xはアップサンプリング前の特徴量
            
            x_up = torch.cat([x_up, skip], dim=1)
        
        # 3. 畳み込み
        return self.conv(x_up)

class AttentionGatedUNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32, 16], high_res_skip_ch=32):
        super().__init__()
        self.b4 = DecoderBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0], attention=True)
        self.b3 = DecoderBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1], attention=True)
        self.b2 = DecoderBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2], attention=True)
        self.b1 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3], attention=True)
        
        # 修正：high_res_skip_ch の受け入れ先を up1 から up3 に移動
        self.up1 = DecoderBlock(decoder_channels[3], 0, decoder_channels[4], attention=False)
        self.up2 = DecoderBlock(decoder_channels[4], 0, 16, attention=False)
        self.up3 = DecoderBlock(16, high_res_skip_ch, 16, attention=False)
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        
        # Deep Supervision用のサブ予測器
        self.deep_sup3 = nn.Conv2d(decoder_channels[1], 1, kernel_size=1) # 1/4スケール
        self.deep_sup2 = nn.Conv2d(decoder_channels[2], 1, kernel_size=1) # 1/2スケール
        self.deep_sup1 = nn.Conv2d(decoder_channels[3], 1, kernel_size=1) # 等倍スケール

    def forward(self, features, high_res_skip=None):
        f0, f1, f2, f3, f4 = features
        x = self.b4(f4, f3)
        x = self.b3(x, f2)
        ds3_out = self.deep_sup3(x)
        
        x = self.b2(x, f1)
        ds2_out = self.deep_sup2(x)
        
        x = self.b1(x, f0)
        ds1_out = self.deep_sup1(x)
        
        # 修正：high_res_skip の結合先を up3 に移動
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x, high_res_skip)
        out = self.final_conv(x)
        
        return out, [ds3_out, ds2_out, ds1_out]

# ==========================================
# 統合モデル (Proposed Hybrid Segmentation Net)
# ==========================================
class HybridSegmentationNet(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, num_heads=4, alpha=0.1, high_res_skip_ch=32):
        super().__init__()
        self.alpha = alpha
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim)
        )
        
        # High-Resolution Skip用の特徴抽出Conv
        self.high_res_proj = nn.Conv2d(in_channels, high_res_skip_ch, kernel_size=1)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SimpleDeformableAttention(embed_dim, num_heads=num_heads, num_points=8)
        self.score_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        self.encoder = timm.create_model(
            'mobilenetv3_large_100', 
            pretrained=True, 
            in_chans=embed_dim, 
            features_only=True
        )
        
        self.decoder = AttentionGatedUNetDecoder(encoder_channels=[16, 24, 40, 112, 960], high_res_skip_ch=high_res_skip_ch)

    def forward(self, x):
        # High-Resolution Skip用の特徴量を保持
        high_res_skip = self.high_res_proj(x)
        
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
        out_mask, deep_sup_masks = self.decoder(encoder_features, high_res_skip)
        
        return out_mask, gate, deep_sup_masks