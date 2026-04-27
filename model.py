import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ==========================================
# 1. Coordinate Attention モジュール
# ==========================================
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(CoordinateAttention, self).__init__()
        # 水平・垂直方向の適応型平均プーリング
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        
        # 方向情報の集約
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 水平方向と垂直方向のアテンションマップを生成
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        # 最終的なゲート値を計算 (B, C, H, W)
        gate = a_h * a_w
        return gate

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
        
        # 出力解像度を入力と一致させる(256px)ための最終アップサンプリング
        self.final_up = DecoderBlock(decoder_channels[3], 0, 16) 
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, features):
        f0, f1, f2, f3, f4 = features
        x = self.b4(f4, f3)
        x = self.b3(x, f2)
        x = self.b2(x, f1)
        x = self.b1(x, f0)
        
        x = self.final_up(x) 
        out = self.final_conv(x)
        return out

# ==========================================
# 3. 統合モデル
# ==========================================
class HybridSegmentationNet(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
        # ImageNetで事前学習済みの標準MobileNetV3を読み込む
        self.encoder = timm.create_model('mobilenetv3_large_100', pretrained=True)
        
        # MobileNetV3_largeの blocks[1] の出力チャンネル数は24
        self.embed_dim = 24 
        
        # Coordinate Attentionへの置き換え
        self.attn = CoordinateAttention(in_channels=self.embed_dim)

        # デコーダの初期化
        self.decoder = LightweightUNetDecoder(encoder_channels=[16, 24, 40, 112, 960])

    def forward(self, x):
        # 【Stem】
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        
        f0 = self.encoder.blocks[0](x)   # (B, 16, H/2, W/2) -> Skip 1
        f1 = self.encoder.blocks[1](f0)  # (B, 24, H/4, W/4) -> Attentionへ
        
        # 【Attention】: Coordinate Attention + α-スケーリング
        raw_gate = self.attn(f1)
        gate = self.alpha + (1.0 - self.alpha) * raw_gate
        
        # 空間情報を保ったまま要素ごとの積（Masking）
        f1_masked = f1 * gate            # (B, 24, H/4, W/4) -> Skip 2
        
        # 【Backbone】
        f2 = self.encoder.blocks[2](f1_masked) # (B, 40, H/8, W/8) -> Skip 3
        
        x_f3 = self.encoder.blocks[3](f2)
        f3 = self.encoder.blocks[4](x_f3)      # (B, 112, H/16, W/16) -> Skip 4
        
        x_f4 = self.encoder.blocks[5](f3)      # (160 channels)
        f4 = self.encoder.blocks[6](x_f4)      # (B, 960, H/32, W/32) -> Skip 5
        
        # 【Decoder】
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
        
        # Dice Loss
        pred_prob = torch.sigmoid(pred_mask)
        intersection = torch.sum(pred_prob * true_mask)
        union = torch.sum(pred_prob) + torch.sum(true_mask)
        l_dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        
        # Sparse Loss (Coordinate Attentionのゲート値全体の平均を利用)
        l_sparse = torch.mean(gate_values)
        
        # 総合 Loss
        total_loss = l_bce + (self.lambda_dice * l_dice) + (self.lambda_sparse * l_sparse)
        
        return total_loss, l_bce, l_dice, l_sparse