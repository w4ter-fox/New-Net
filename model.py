import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ==========================================
# 1. 損失関数 (Deep Supervision対応)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, lambda_sparse=0.1, lambda_dice=1.0, lambda_deep_sup=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_sparse = lambda_sparse
        self.lambda_dice = lambda_dice
        self.lambda_deep_sup = lambda_deep_sup

    def forward(self, pred_mask, true_mask, gate_values, deep_sup_masks=None):
        l_seg_final = self.calc_seg_loss(pred_mask, true_mask)
        l_seg_deep = 0.0
        if deep_sup_masks is not None:
            for mask in deep_sup_masks:
                target_rescaled = F.interpolate(true_mask, size=mask.shape[2:], mode='nearest')
                l_seg_deep += self.calc_seg_loss(mask, target_rescaled)
            l_seg_deep /= len(deep_sup_masks)

        l_seg = l_seg_final + (self.lambda_deep_sup * l_seg_deep)
        l_sparse = torch.mean(gate_values)
        return l_seg + (self.lambda_sparse * l_sparse), l_seg, l_sparse

    def calc_seg_loss(self, pred, target):
        l_bce = self.bce(pred, target)
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * target).sum(dim=(2, 3))
        uni = pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - (2. * inter + 1e-5) / (uni + 1e-5)
        return l_bce + self.lambda_dice * dice.mean()

# ==========================================
# 2. Coordinate Attention (v4 高速化の要)
# ==========================================
class CoordinateAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_w * a_h

# ==========================================
# 3. デコーダ構成部品
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_ag=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.ag = AttentionGate(in_ch, skip_ch, skip_ch // 2) if use_ag else None
        # Depthwise Separable Conv でさらに軽量化
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, in_ch + skip_ch, 3, padding=1, groups=in_ch + skip_ch, bias=False),
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if self.ag: skip = self.ag(g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class HighSpeedDecoder(nn.Module):
    # 【追加】use_gateパラメータを受け取り、デコーダ内のAttentionGateを一括管理
    def __init__(self, enc_chs, high_res_ch=32, use_gate=True):
        super().__init__()
        self.b4 = DecoderBlock(enc_chs[4], enc_chs[3], 256, use_ag=use_gate)
        self.b3 = DecoderBlock(256, enc_chs[2], 128, use_ag=use_gate)
        self.b2 = DecoderBlock(128, enc_chs[1], 64, use_ag=use_gate)
        self.b1 = DecoderBlock(64, enc_chs[0], 32, use_ag=use_gate)
        
        # 32x32 -> 256x256 へ戻すための3段階アップサンプリング
        self.up1 = DecoderBlock(32, 0, 16, use_ag=False) # 32x32 -> 64x64
        self.up2 = DecoderBlock(16, 0, 16, use_ag=False) # 64x64 -> 128x128
        self.up3 = DecoderBlock(16, high_res_ch, 16, use_ag=False) # 128x128 -> 256x256
        
        self.final_conv = nn.Conv2d(16, 1, 1)
        
        # Deep Supervision用
        self.ds3 = nn.Conv2d(128, 1, 1)
        self.ds2 = nn.Conv2d(64, 1, 1)

    def forward(self, features, high_res_skip):
        f0, f1, f2, f3, f4 = features
        x = self.b4(f4, f3)
        x = self.b3(x, f2); ds3 = self.ds3(x)
        x = self.b2(x, f1); ds2 = self.ds2(x)
        x = self.b1(x, f0)  # ここで解像度は 32x32
        
        # up1, up2, up3 を順番に通過させる
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x, high_res_skip)
        
        return self.final_conv(x), [ds3, ds2]

# ==========================================
# 4. Hybrid Net v4 (Proposed High-Speed)
# ==========================================
class HybridSegmentationNet(nn.Module):
    # 【追加】初期化時に機能のon/offを受け取る
    def __init__(self, alpha=0.1, use_coord_attn=True, use_gate=True):
        super().__init__()
        self.alpha = alpha
        self.use_coord_attn = use_coord_attn
        self.use_gate = use_gate
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.high_res_proj = nn.Conv2d(3, 32, 1)
        self.attn = CoordinateAttention(64, 64) # 高速なCoordAttへ変更
        self.score_proj = nn.Conv2d(64, 1, 1)
        
        self.encoder = timm.create_model('mobilenetv3_large_100', pretrained=True, features_only=True, in_chans=64)
        # 【追加】use_gateフラグをデコーダへ渡す
        self.decoder = HighSpeedDecoder(enc_chs=[16, 24, 40, 112, 960], use_gate=self.use_gate)

    def forward(self, x):
        hr_skip = self.high_res_proj(x)
        x_stem = self.stem(x)
        
        # 【追加】Coordinate Attention の ON/OFF
        if self.use_coord_attn:
            z_attn = self.attn(x_stem)
        else:
            z_attn = x_stem
        
        # 【追加】Soft Gate の ON/OFF
        if self.use_gate:
            gate = self.alpha + (1.0 - self.alpha) * torch.sigmoid(self.score_proj(z_attn))
            x_masked = z_attn * gate
        else:
            # ゲート無効時は恒等写像として扱い、Loss計算エラーを防ぐために1.0のダミーテンソルを生成
            gate = torch.ones(z_attn.shape[0], 1, z_attn.shape[2], z_attn.shape[3], device=z_attn.device)
            x_masked = z_attn
        
        enc_feats = self.encoder(x_masked)
        out, ds_list = self.decoder(enc_feats, hr_skip)
        
        return out, gate, ds_list